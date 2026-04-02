# MiniMind - 从零构建 LLM

> 🌿 一个从零开始手写的 Mini LLM，支持预训练、SFT、LoRA、DPO 全流程

---

## 目录

- [项目结构](#项目结构)
- [核心架构讲解](#核心架构讲解)
  - [整体数据流](#整体数据流)
  - [RMSNorm - 归一化](#rmsnorm--归一化)
  - [RoPE + YARN - 位置编码](#rope--yarn---位置编码)
  - [GQA - 分组查询注意力](#gqa---分组查询注意力)
  - [FFN (SwiGLU) - 前馈网络](#ffn-swiglu---前馈网络)
  - [MiniMindBlock - Transformer 层](#minimindblock---transformer-层)
  - [MiniMindForCausalLM - 完整模型](#minimindforcausallm---完整模型)
- [开始预训练](#开始预训练)
  - [环境准备](#环境准备)
  - [数据准备](#数据准备)
  - [启动训练](#启动训练)
  - [断点续训](#断点续训)
- [常见问题](#常见问题)

---

## 项目结构

```
Minimind/
├── model/
│   └── model.py          # 模型核心：MiniMindForCausalLM
├── dataset/
│   └── lm_dataset.py      # 数据集加载
├── trainer/
│   ├── trainer_pretrain.py   # 预训练入口
│   └── trainer_utrls.py      # 工具函数（SFT/LoRA等）
├── checkpoints/          # 训练保存的权重
├── out/                  # 输出权重
└── dataset/
    └── pretrain_hq.jsonl   # 预训练数据
```

---

## 核心架构讲解

### 整体数据流

```
输入文本 ("hello")
    ↓
Tokenizer (分词) → token_ids
    ↓
Embedding (词嵌入) → hidden_states
    ↓
Transformer Layer × K (K层堆叠)
    ↓
RMSNorm (最终归一化)
    ↓
LM Head (线性层)
    ↓
SoftMax → 概率分布
    ↓
输出文本 ("hello + world")
```

---

### RMSNorm - 归一化

**RMSNORM** = Root Mean Square Layer Normalization，是 LayerNorm 的简化加速版。

**对比：**

| 步骤 | LayerNorm | RMSNORM |
|------|-----------|---------|
| 计算均值 μ | ✅ | ❌ |
| 计算方差 | ✅ | ❌ |
| 计算 RMS | ❌ | ✅ `√(mean(x²))` |
| 归一化 | `(x - μ) / √(σ² + ε)` | `x / (RMS + ε)` |

**为什么更快？** 少算一次均值，少一次减法操作。

**公式：**
```
RMS = √(mean(x²) + ε)
x_norm = x / RMS
output = γ * x_norm  (可学习缩放)
```

**在 MiniMind 中的位置：** Pre-Norm 设计，每个子层前都做一次 RMSNorm
```python
hidden_states = residual + self.self_attn(self.input_layernorm(hidden_states))
hidden_states = residual + self.mlp(self.post_attn_layernorm(hidden_states))
```

---

### RoPE + YARN - 位置编码

#### RoPE（旋转位置编码）

**核心思想：** 不是把位置加到 token 上，而是"旋转"进向量里。

```
位置0 → 不旋转 (0°)
位置1 → 旋转 θ 角度
位置2 → 旋转 2θ 角度
...
位置i → 旋转 i × θ 角度
```

**公式：**
```
q' = q * cos(θ) + rotate_half(q) * sin(θ)
k' = k * cos(θ) + rotate_half(k) * sin(θ)

其中 rotate_half 把向量后半取负：
[x1, x2, x3, x4] → [-x3, -x4, x1, x2]
```

**优势：**
- 外推性好（训练2048，推理可更长）
- 保留相对位置信息
- 不需要额外位置参数

#### YARN（YaRN 缩放）

解决超长上下文（>32k）外推问题，通过频率缩放让长序列效果更好。

```python
rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,          # 位置扩大16倍
    "original_max_position_embeddings": 2048,
}
# 训练长度：2048
# 推理长度：2048 × 16 = 32768
```

**原理：** 低频维度缩放多，高频维度缩放少，保持短距离依赖不变。

---

### GQA - 分组查询注意力

**GQA** = Grouped Query Attention，是 MHA 和 MQA 的折中。

**三种注意力机制对比：**

| 类型 | Q头数 | K头数 | V头数 | 显存 | 速度 |
|------|-------|-------|-------|------|------|
| MHA | 8 | 8 | 8 | 大 | 慢 |
| MQA | 8 | 1 | 1 | 小 | 快 |
| **GQA** | 8 | **2** | **2** | 中 | 中 |

**核心：** K/V 头数 < Q 头数，通过 `repeat_kv` 复制来匹配维度

```python
n_rep = num_attention_heads // num_key_value_heads  # 8 // 2 = 4
# 原来: [K0, K1]
# 复制后: [K0, K0, K0, K0, K1, K1, K1, K1]
```

**为什么省显存？** K/V Cache 是推理时的瓶颈，减少 K/V 头数直接减少缓存大小。

**完整 Attention 计算流程：**
```
输入x
    ↓
RMSNorm (Pre-Norm)
    ↓
Linear → Q [bsz, seq, 8, 64], K [bsz, seq, 2, 64], V [bsz, seq, 2, 64]
    ↓
repeat_kv(K, 4), repeat_kv(V, 4) → 都变成 [bsz, seq, 8, 64]
    ↓
RoPE 位置编码 (Q 和 K)
    ↓
transpose → [bsz, 8, seq, 64]
    ↓
Q × K^T / √head_dim → Attention Scores
    ↓
+ Causal Mask (上三角=-∞)
    ↓
SoftMax → 归一化注意力权重
    ↓
× V → Context
    ↓
O_proj → 输出
```

**Causal Mask（因果掩码）：** 防止看到未来 token
```
      Q0  Q1  Q2  Q3
K0 [   0  -∞  -∞  -∞ ]
K1 [   0    0  -∞  -∞ ]
K2 [   0    0    0  -∞ ]
K3 [   0    0    0    0 ]
```

---

### FFN (SwiGLU) - 前馈网络

**FFN** = Feed-Forward Network，给 token 增加非线性表达能力。

**参数量：** 占整个 Transformer 的 ~2/3

**结构：** 升维 → 激活 → 降维
```
输入(512维)
    ↓
升维: Linear → 512 → 2048
    ↓
激活函数 (SiLU)
    ↓
降维: Linear → 2048 → 512
    ↓
输出(512维)
```

**SwiGLU 公式：**
```
FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```

**流程：**
```
x 分两路：
┌──────────────┐    ┌──────────────┐
│  up_proj    │    │  gate_proj   │
│    x        │    │    x         │
└──────────────┘    └──────────────┘
       ↓                  ↓
      x             SiLU → 激活
       ↓                  ↓
       └────── ⊙ ────────┘ ← 逐元素相乘
                  ↓
             down_proj → 降维
                  ↓
              dropout
```

**SiLU 激活函数：**
```
SiLU(x) = x × sigmoid(x) = x · σ(x)
```
- x→-∞ 时，SiLU → -1/e ≈ -0.368（不完全抑制）
- x=0 时，SiLU = 0
- 比 ReLU 更平滑，梯度更稳定

---

### MiniMindBlock - Transformer 层

每个 MiniMindBlock = 一个完整的 Transformer 层，包含两次残差连接：

```
┌─────────────────────────────────────┐
│           MiniMindBlock              │
│                                      │
│  hidden_states                        │
│       ↓                               │
│  ┌──────────────────────────────┐    │
│  │     RMSNorm (Pre-Norm)        │    │
│  └──────────────────────────────┘    │
│       ↓                               │
│  ┌──────────────────────────────┐    │
│  │      Attention (GQA)         │    │
│  │  Q, K, V → RoPE → Attn       │    │
│  └──────────────────────────────┘    │
│       ↓                               │
│  ⊕ residual + hidden_states    │ ← 残差1
│                                      │
│  ┌──────────────────────────────┐    │
│  │     RMSNorm (Pre-Norm)        │    │
│  └──────────────────────────────┘    │
│       ↓                               │
│  ┌──────────────────────────────┐    │
│  │        FFN (SwiGLU)           │    │
│  └──────────────────────────────┘    │
│       ↓                               │
│  ⊕ residual + hidden_states    │ ← 残差2
│                                      │
│  hidden_states (输出给下一层)         │
└─────────────────────────────────────┘
```

**Pre-Norm vs Post-Norm：**
- Pre-Norm（MiniMind用）：先 Norm 再 Attention/FFN → 训练更稳定
- Post-Norm（原版）：先 Attention/FFN 再 Norm

**残差连接的作用：**
- 梯度"近道"：可直接传回底层，不用每层都穿过
- 防止退化：即使 MLP/Attn 学坏了，原始信息还在
- 帮助优化：让深层网络更容易训练

---

### MiniMindForCausalLM - 完整模型

**整体架构：**
```
Embedding → Block1 → Block2 → ... → Block8 → RMSNorm → LM Head → SoftMax
  (tok_emb)                                          (lm_head)
```

**核心配置：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 512 | 隐藏层维度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `num_attention_heads` | 8 | Query 头数 |
| `num_key_value_heads` | 2 | K/V 头数（GQA） |
| `vocab_size` | 6400 | 中文词表大小 |
| `max_position_embeddings` | 32768 | YARN 扩展后最大长度 |
| `rope_theta` | 1000000 | RoPE 基础频率 |
| `head_dim` | 64 | 每个头的维度 |

---

## 开始预训练

### 环境准备

```bash
# 克隆项目
cd Minimind

# 使用 uv 创建虚拟环境（项目自带 .venv）
source .venv/Scripts/activate  # Windows
# 或 source .venv/bin/activate  # Linux

# 确认依赖已安装
pip install torch transformers datasets tqdm
```

### 数据准备

项目使用 HuggingFace `datasets` 库惰性加载 JSONL 格式数据。

**数据格式（JSONL，每行一条）：**
```json
{"text": "今天天气很好，适合出去散步。"}
{"text": "人工智能正在改变我们的生活方式。"}
```

**准备数据文件：**
```bash
# 将数据放到 dataset/ 目录下
# 确保文件路径正确（默认查找 ../dataset/pretrain_hq.jsonl）
```

### 启动训练

```bash
# 进入项目目录
cd Minimind

# 激活环境
source .venv/Scripts/activate

# 运行预训练（单卡）
python -m trainer.trainer_pretrain \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 512 \
    --data_path ../dataset/pretrain_hq.jsonl

# 启用 wandb 实验跟踪
python -m trainer.trainer_pretrain \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --use_wandb \
    --wandb_project MiniMind-Pretrain

# 使用 MoE 架构
python -m trainer.trainer_pretrain \
    --use_moe 1 \
    --epochs 1 \
    --batch_size 32
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数 |
| `--batch_size` | 32 | batch size |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--max_seq_len` | 512 | 最大序列长度 |
| `--hidden_size` | 512 | 隐藏层维度 |
| `--num_hidden_layers` | 8 | 层数 |
| `--use_moe` | 0 | 是否用 MoE (0/1) |
| `--device` | cuda:0 | 训练设备 |
| `--log_interval` | 100 | 日志打印间隔 |
| `--save_interval` | 100 | 保存间隔 |
| `--use_wandb` | - | 启用 wandb |

### 断点续训

```bash
# 自动检测并续训
python -m trainer.trainer_pretrain \
    --from_resume 1 \
    --epochs 2

# 从指定权重继续训练
python -m trainer.trainer_pretrain \
    --from_weight pretrain \
    --epochs 2
```

---

## 常见问题

**Q: 没有 GPU 能跑吗？**
A: CPU 能跑，但非常慢（可能几天）。建议用 GPU 服务器，AutoDL 上 3090 每小时不到 2 块钱。

**Q: 显存不够怎么办？**
A: 减小 `--max_seq_len`，或减小 `--hidden_size` / `--num_hidden_layers`。

**Q: 训练到一半断了？**
A: 加 `--from_resume 1` 自动检测并续训。

**Q: 怎么查看训练loss？**
A: 启用 `--use_wandb` 在 wandb dashboard 看曲线。

---

*🌿 项目学习笔记整理自 Yeeko 的学习记录*
