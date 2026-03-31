from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional,Tuple
from torch.nn import functional as F
from .activation_functions import ACT2FN

# RMSNORM
# inherit nn.Module
class RMSNorm(nn.Module):
    #dim: 维度；eps：norm所需极小值
    def _init_(self, dim: int, eps: 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    #norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim = True)+self.eps) * x
        

    #forward
    def forward(self,x):
        return self.weight*self._norm(x.float().type_as(x))

# 1. 预计算 YaRN 频率
def precompute_freqs_cis(dim: int, end: int = 32*1024, rope_base: float = 10000.0, rope_scaling: Optional[dict] = None):
    # 修正语法错误，移除多余的切片
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    attn_factor = 1.0
    
    if rope_scaling is not None:
        orig_max = rope_scaling['original_max_position_embeddings']
        factor = rope_scaling['factor']
        beta_fast = rope_scaling['beta_fast']
        beta_slow = rope_scaling['beta_slow']

        # 推断的长度大于训练长度，使用缩放
        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            # 划分高低维度边界
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

            # 计算缩放因子，处理平滑过渡
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0,
                1,
            )

            # 对频率进行线性插值缩放
            freqs = freqs * (1 - ramp + ramp / factor)
            
            # 【补充】YaRN 的 Attention 缩放因子 (mscale)
            # 作用是修正拓展上下文后的注意力分布熵变
            mscale = 0.1 * math.log(factor) + 1.0
            attn_factor = mscale # 将乘在 cos 和 sin 上

    # 【修复缩进】无论是否缩放，都必须执行以下生成位置索引的逻辑
    t = torch.arange(end, device=freqs.device).float()

    # 计算外积 (end, dim//2)
    freqs = torch.outer(t, freqs).float()

    # 拼接并应用 attn_factor (end, dim)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=1) * attn_factor

    return freqs_cos, freqs_sin

# 2. 应用 RoPE
def apply_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 旋转一半的维度: [a, b, c, d] -> [-c, -d, a, b] 
    # (假设 LLaMA 风格的 interleave 方式，前半和后半分别对应)
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1
        )
    
    # 【修复】必须根据 position_ids 提取对应的位置！
    if position_ids is not None:
        # cos 形状变为 (batch_size, seq_len, dim)
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        # 如果没有传入 position_ids，默认按 q 的 seq_len 截断
        # 假设 q 的形状为 (batch, seq_len, num_heads, head_dim)
        seq_len = q.shape[1] 
        cos = cos[:seq_len]
        sin = sin[:seq_len]

    # 给 cos 和 sin 增加维度以匹配 q, k (通常在 head 的维度扩充)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 核心公式：x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
    

# 复制KV (GQA核心逻辑)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 获取四维张量 [bsz, seq_len, n_kv_heads, head_dim]
    bs, slen, num_key_value_heads, head_dim = x.shape
    # 如果重复维度为1，不需要重复复制
    if n_rep == 1:
        return x
    # 在第四个维度插入新维度并展开，最后reshape合并头
    return x[:, :, :, None, :].expand(
        bs, slen, num_key_value_heads, n_rep, head_dim
    ).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)

# Attention模块
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()

        # 修复：初始化 kv_heads 逻辑
        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        
        assert args.num_attention_heads % self.num_key_value_heads == 0, 'num_attention_heads必须被num_key_value_heads整除'

        self.n_local_heads = args.num_attention_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 修复：== 改为 =，并修正 k_proj 和 v_proj 的输出维度适配 GQA
        self.q_proj = nn.Linear(args.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出的投影层
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, args.hidden_size, bias=False)

        # dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # flash attention 开关
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None # 修复类型和默认值
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, seq_len, _ = x.shape
        
        # 修复：拼写错误 elf -> self
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 拆分成多个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        # 提取 RoPE
        cos, sin = position_embedding
        # 修复：sq 改为 xq，修复切片传参语法
        xq, xk = self.apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        
        # KV Cache 逻辑
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        past_kv = (xk, xv) if use_cache else None

        # 修复：千万不能用 {} Set，必须用 () Tuple！否则每次返回顺序是乱的
        xq, xk, xv = (
            xq.transpose(1, 2), # [bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 进行attention计算
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 修复：is_casual -> is_causal
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=None, # 如果是标准 causal，通常让底层自己处理 mask
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # 自己编写 Attention
            # 修复：加上等于号 =
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Causal Mask (当前只支持单向因果)
            # 修复：self.float('-inf') -> float('-inf')，unsqueeze(9) -> unsqueeze(0).unsqueeze(0)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

            # 外部传入的 Mask 处理 (例如 Padding Mask)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 修复：mask 为 0 的地方应当减去极大的值，而不是加上
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
        
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 拼接头，输出投影，返回
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 修复：self,p_proj -> self.o_proj
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
    

#FFN
class FeedForward(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)
    #升维linear
        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
    #降维
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
    #门控
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
    #dropout
        self.dropout=nn.Dropout(args.dropout)
    #激活函数SILU
        self.act_fn=ACT2FN[args.hidden_act]

    #向前传播
    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))

#拼接GQA和FFN
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id:int, args:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(args)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = FeedForward(args)

    #向前传播
    def forward(self, hidden_states,position_embeddings,past_key_value=None,
                use_cache=False,attention_mask=None):
        #残差化
        residual = hidden_states
        hidden_states,present_key_value=self.self_attn(
            self.input_layernorm(hidden_states),position_embeddings,past_key_value,
            use_cache,attention_mask
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attn_layernorm(hidden_states))

        return hidden_states, present_key_value