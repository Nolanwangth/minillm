# MiniLLM — 从零构建的 GPT 语言模型

纯 PyTorch 实现的完整 LLM，涵盖预训练和 SFT 微调，每一行代码都有中文注释。
专为学习设计：麻雀虽小，五脏俱全。

## 完整训练流程

```
第一阶段：预训练 (Pre-training)
  喂大量文本 → 学语言规律 → 得到基础模型

第二阶段：监督微调 (SFT)
  喂问答对 → 只对回答部分算 loss → 变成对话模型
```

## 架构

```
输入 token IDs
      │
      ▼
Token Embedding + Positional Embedding
      │
      ▼
┌──────────────────────────┐
│   TransformerBlock × N   │
│  ┌────────────────────┐  │
│  │  LayerNorm         │  │
│  │  因果多头自注意力   │  │  ← 每个 token 只能看左边
│  │  + 残差连接        │  │
│  ├────────────────────┤  │
│  │  LayerNorm         │  │
│  │  前馈网络 (FFN)    │  │  ← 每个位置独立处理，4× 扩展
│  │  + 残差连接        │  │
│  └────────────────────┘  │
└──────────────────────────┘
      │
      ▼
LayerNorm → LM Head → logits (vocab_size)
      │
      ▼
预训练：对所有 token 算交叉熵 loss
SFT：只对 assistant 回复部分算 loss
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `model.py` | 完整 Transformer：Attention、FFN、残差、Weight Tying |
| `config.py` | 超参数预设（nano / small） |
| `tokenizer.py` | 字节级分词器（vocab_size=256） |
| `dataset.py` | 滑动窗口 next-token 数据集 |
| `train.py` | 预训练脚本：LR warmup/cosine decay、checkpoint |
| `sft_data.py` | SFT 数据格式、内置问答集、loss mask 逻辑 |
| `sft.py` | SFT 微调脚本：只对回复部分算 loss |
| `generate.py` | 从 checkpoint 推理生成文字 |
| `download_data.py` | 下载 TinyShakespeare / TinyStories |

## 快速开始

```bash
# 1. 创建环境
conda create -n minillm_env python=3.11 -y
conda activate minillm_env
pip install -r requirements.txt

# 2. 下载训练数据
python download_data.py shakespeare

# 3. 预训练
python train.py --data data/shakespeare.txt --epochs 200

# 4. SFT 微调（在预训练基础上）
python sft.py --checkpoint checkpoints/best.pt

# 5. 推理
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
python generate.py --checkpoint checkpoints/sft_best.pt --prompt "[USER]什么是注意力机制？[ASST]"
```

## 核心概念

### 因果自注意力
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```
因果掩码（下三角矩阵）让位置 t 只能看到 ≤ t 的位置，
保证模型不能"作弊"看未来。

### 残差连接
```python
x = x + self.attn(self.ln1(x))
x = x + self.ff(self.ln2(x))
```
给梯度提供"高速公路"，没有它深层网络无法训练。

### SFT 的核心：loss mask
```python
# y 中 user 部分 = -100，PyTorch 自动跳过这些位置
loss = cross_entropy(logits, y, ignore_index=-100)
```
只对 assistant 回复部分计算 loss，这一行就是 SFT 和预训练的核心区别。

### Weight Tying
Token Embedding 和 LM Head 共享同一组权重，
减少参数量，同时让语义相近的 token 有相近的向量。

### Temperature & Top-k
```
temperature < 1.0 → 输出更保守、重复
temperature > 1.0 → 输出更随机、有创意
top_k = 40       → 只从最高概率的 40 个 token 里采样
```

## 模型大小

| 预设 | 参数量 | n_embd | n_heads | n_layers | block_size |
|------|--------|--------|---------|----------|------------|
| nano | ~222K | 64 | 4 | 4 | 128 |
| small | ~1.5M | 256 | 8 | 6 | 256 |

## 调试

项目自带 `.vscode/launch.json`，用 Cursor 或 VS Code 打开文件夹后，
`Run & Debug` 面板直接选配置按 F5：

| 配置名 | 作用 |
|--------|------|
| 训练 (nano, 内置语料) | 快速验证，几秒跑完 |
| 训练 (Shakespeare) | 真实数据训练 |
| SFT 微调 | 在预训练基础上微调 |
| 推理生成 | 加载 checkpoint 生成文字 |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — 本项目的灵感来源
- [LLaMA 3 论文](https://arxiv.org/abs/2407.21783) — 工业级 LLM 的进阶阅读
