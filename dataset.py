"""
数据集工具。

语言模型的训练目标叫做"下一个 token 预测"（Next Token Prediction）：
给模型看前面的 token，让它预测下一个是什么。

具体做法是滑动窗口：
  - 把整个文本 tokenize 成一个长数字列表
  - 每个训练样本是一段连续的 (block_size + 1) 个 token
  - 输入 x = tokens[i : i + block_size]        （前 block_size 个）
  - 标签 y = tokens[i+1 : i + block_size + 1]  （后 block_size 个，即 x 整体右移一位）

举例：文本 "hello"，block_size=3
  tokens = [104, 101, 108, 108, 111]
  样本0:  x=[104,101,108]  y=[101,108,108]
           位置0: 看到104('h')  → 预测101('e')
           位置1: 看到104,101  → 预测108('l')
           位置2: 看到104,101,108 → 预测108('l')

这样一个样本就能同时训练 block_size 个预测任务，非常高效。
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int):
        # 把整个文本编码成一个大的整数列表，存成 PyTorch 张量
        ids = tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # 每个样本需要 block_size+1 个 token（x 用 block_size 个，y 多 1 个）
        # 所以总样本数 = 总 token 数 - block_size
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        # 取出一段长度为 block_size+1 的片段
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # 去掉最后一个 → 输入序列
        y = chunk[1:]   # 去掉第一个   → 标签序列（x 向右移一位）
        return x, y


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
