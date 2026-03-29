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
from typing import Optional

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