"""
Dataset utilities.

We use a simple sliding-window approach:
  - Tokenize the entire corpus into one long list of IDs
  - Each training sample is a window of (block_size + 1) tokens
  - Input  x = tokens[i : i + block_size]
  - Target y = tokens[i+1 : i + block_size + 1]  (shifted by 1)

This means the model learns to predict the next token at every position —
that's the "language modelling" objective (also called CLM or next-token prediction).
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int):
        ids = tokenizer.encode(text)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # Each sample needs block_size + 1 tokens (input + label)
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]    # target tokens (next-token labels)
        return x, y


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
