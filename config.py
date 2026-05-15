"""
Model configuration. All hyperparameters live here.

Presets:
  - nano:  ~50K params  — trains in seconds on CPU, good for debugging
  - small: ~1.5M params — needs GPU or patience, better quality
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocabulary & sequence
    vocab_size: int = 256      # character-level: 256 byte values
    block_size: int = 128      # maximum context length (tokens)

    # Architecture
    n_embd: int = 64           # embedding / hidden dimension
    n_heads: int = 4           # number of attention heads (n_embd must be divisible)
    n_layers: int = 4          # number of Transformer blocks

    # Regularisation
    dropout: float = 0.1


# Ready-made presets
NANO = ModelConfig(n_embd=64,  n_heads=4, n_layers=4,  block_size=128)
SMALL = ModelConfig(n_embd=256, n_heads=8, n_layers=6,  block_size=256)
