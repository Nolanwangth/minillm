"""
模型超参数配置。

所有控制模型大小和行为的参数都在这里，
改这里的数字就能得到不同大小的模型。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # ── 词表和序列 ────────────────────────────────────────────────────────────
    vocab_size: int = 256   # 词表大小。字节级分词器固定是 256
    block_size: int = 128   # 最大上下文长度（token 数）。
                            # 模型一次最多能"看到"这么多 token

    # ── 模型架构 ──────────────────────────────────────────────────────────────
    n_embd: int = 64        # embedding 维度，也是每个 Transformer 层的隐藏维度。
                            # 越大模型越有表达能力，但参数量和计算量也越大

    n_heads: int = 4        # 注意力头数。n_embd 必须能被 n_heads 整除。
                            # 每个头的维度 = n_embd / n_heads = 16

    n_layers: int = 4       # Transformer Block 的层数，也叫"深度"。
                            # GPT-3 有 96 层，我们用 4 层

    # ── 正则化 ────────────────────────────────────────────────────────────────
    dropout: float = 0.1    # Dropout 概率。训练时随机把 10% 的神经元置零，
                            # 防止过拟合。推理时自动关闭


# 预设配置，直接拿来用
NANO  = ModelConfig(n_embd=64,  n_heads=4, n_layers=4, block_size=128)
# nano 约 222K 参数，CPU 上几分钟就能训练出点效果，适合学习和调试

SMALL = ModelConfig(n_embd=256, n_heads=8, n_layers=6, block_size=256)
# small 约 1.5M 参数，需要 GPU 或耐心等待，生成质量更好
