"""
SFT 数据格式定义和内置问答数据集。

SFT（监督微调）和预训练的核心区别只有一个：
  - 预训练：对所有 token 计算 loss（学习语言规律）
  - SFT：只对 assistant 的回复部分计算 loss（学习怎么回答问题）

为什么要屏蔽 user 部分的 loss？
  因为 user 的问题是"条件"，不是模型需要学会生成的东西。
  如果对问题也算 loss，相当于在教模型"自言自语地提问"，没有意义。

对话格式：
  [USER] 用户的问题 [ASST] 模型的回答 [END]

训练时：
  输入 x = [USER] 问题 [ASST] 回答 [END]
  标签 y = 同上，但 [USER]....[ASST] 这段的 loss 被 mask 掉（设为 -100）
  PyTorch 的 cross_entropy 默认忽略值为 -100 的位置
"""

import json
import torch
from torch.utils.data import Dataset
from tokenizer import ByteTokenizer

# 特殊标记，用普通字符串表示，字节级分词器会把它们编码成字节序列
USER_TOKEN = "[USER]"
ASST_TOKEN = "[ASST]"
END_TOKEN  = "[END]"

# loss mask 用的忽略值，PyTorch cross_entropy 会自动跳过这些位置
IGNORE_INDEX = -100


# ── 内置问答数据集（关于 Transformer 和 LLM）──────────────────────────────────
BUILTIN_DATA = [
    {
        "user": "什么是 Transformer？",
        "assistant": "Transformer 是一种深度学习架构，由 Google 在 2017 年的论文《Attention Is All You Need》中提出。它的核心是自注意力机制，让模型能直接建模序列中任意两个位置之间的关系，不像 RNN 需要逐步传递信息。现代所有主流大语言模型（GPT、LLaMA、Claude）都基于 Transformer。"
    },
    {
        "user": "什么是自注意力机制？",
        "assistant": "自注意力让序列中的每个 token 都能'看到'其他所有 token，并根据相关性加权融合它们的信息。核心公式是 Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V。Q 是查询，K 是键，V 是值。Q 和 K 的点积衡量两个 token 的相关性，softmax 转成权重，再用权重对 V 加权求和。"
    },
    {
        "user": "什么是因果掩码？为什么需要它？",
        "assistant": "因果掩码是一个下三角矩阵，用于屏蔽注意力矩阵中右上角的位置。这样每个 token 只能看到它自己和它左边的 token，无法看到未来的 token。这对语言模型很关键：训练时如果能看到答案再预测，模型就学不到真正的语言规律。推理时按这种方式生成，模型才能自回归地一个个生成 token。"
    },
    {
        "user": "残差连接有什么用？",
        "assistant": "残差连接就是 x = x + sublayer(x)，把子层的输入直接加到输出上。它有两个重要作用：第一，为梯度提供一条'高速公路'，让梯度能直接从深层流回浅层，解决深度网络的梯度消失问题；第二，即使某层学到的变换很小甚至是零，信息也能通过残差路径无损传递。ResNet 和所有现代 Transformer 都依赖残差连接。"
    },
    {
        "user": "预训练和 SFT 有什么区别？",
        "assistant": "预训练是在大量无标注文本上训练，目标是预测下一个 token，让模型学会语言规律和世界知识。SFT（监督微调）是在人工标注的问答对上继续训练，只对 assistant 的回复部分计算 loss，教模型按照特定格式回答问题。预训练让模型'懂语言'，SFT 让模型'会对话'。GPT-3 经过 SFT 变成 InstructGPT，再经过 RLHF 变成 ChatGPT。"
    },
    {
        "user": "什么是 temperature？",
        "assistant": "Temperature 是控制生成随机性的参数。模型输出 logits 后，在 softmax 之前除以 temperature。temperature=1 时按原始概率采样；temperature<1 时概率分布更尖锐，高概率的 token 更容易被选中，输出更保守重复；temperature>1 时分布更平坦，低概率 token 也有机会被选中，输出更随机有创意。ChatGPT 的'创造力'滑块本质上就是在调 temperature。"
    },
    {
        "user": "什么是 top-k 采样？",
        "assistant": "Top-k 采样是指每次生成时只从概率最高的 k 个 token 里随机选一个，把其他 token 的概率设为 0。这样能避免采样到概率很低的奇怪 token，同时保留一定的随机性。k=1 就是贪心解码（每次选最高概率），k=词表大小 就是完全随机采样。实际使用中 k=40 左右效果不错。"
    },
    {
        "user": "LayerNorm 的作用是什么？",
        "assistant": "LayerNorm（层归一化）把每个样本的特征归一化到均值为 0、方差为 1，然后用可学习的参数做缩放和平移。它的作用是稳定训练：深层网络中，每层的输入分布会随训练不断变化（内部协变量偏移），导致训练不稳定。LayerNorm 把分布拉回到一个合理的范围，让每层的输入更可预测，梯度更稳定。"
    },
]


def format_conversation(user: str, assistant: str) -> str:
    """把一条问答对格式化成模型输入的字符串。"""
    return f"{USER_TOKEN}{user}{ASST_TOKEN}{assistant}{END_TOKEN}"


class SFTDataset(Dataset):
    """
    SFT 数据集。

    关键点：只对 assistant 回复部分计算 loss。
    做法：把 user 部分对应的标签设为 IGNORE_INDEX (-100)，
    PyTorch 的 cross_entropy 会自动跳过这些位置。
    """

    def __init__(self, data: list[dict], tokenizer: ByteTokenizer, block_size: int):
        self.samples = []
        self.block_size = block_size

        for item in data:
            user_text = item["user"]
            asst_text = item["assistant"]

            # 编码各个部分
            prefix = tokenizer.encode(USER_TOKEN + user_text + ASST_TOKEN)
            response = tokenizer.encode(asst_text + END_TOKEN)
            full = prefix + response

            # 截断到 block_size
            full = full[:block_size]

            # 构造 input 和 label
            input_ids = torch.tensor(full, dtype=torch.long)

            # label 默认全是 IGNORE_INDEX（不计算 loss）
            labels = torch.full_like(input_ids, IGNORE_INDEX)

            # 只有 response 部分的 label 是真实 token ID
            response_start = min(len(prefix), len(full))
            labels[response_start:] = input_ids[response_start:]

            self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_jsonl(path: str) -> list[dict]:
    """
    从 JSONL 文件加载问答数据。

    每行一个 JSON 对象，格式：{"user": "...", "assistant": "..."}
    例如：
        {"user": "什么是神经网络？", "assistant": "神经网络是..."}
        {"user": "解释反向传播", "assistant": "反向传播是..."}
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
