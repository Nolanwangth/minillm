# MiniLLM — Transformer LLM from Scratch

A tiny but complete GPT-style language model built from scratch in pure PyTorch.
Designed for learning: every component is explained with comments.

## Architecture

```
Input tokens
    │
    ▼
Token Embedding + Positional Embedding
    │
    ▼
┌─────────────────────────┐
│   TransformerBlock × N  │
│  ┌───────────────────┐  │
│  │  LayerNorm        │  │
│  │  CausalMHA        │  │  ← Multi-Head Self-Attention (causal mask)
│  │  + Residual       │  │
│  ├───────────────────┤  │
│  │  LayerNorm        │  │
│  │  FeedForward      │  │  ← 2-layer MLP, 4× expansion
│  │  + Residual       │  │
│  └───────────────────┘  │
└─────────────────────────┘
    │
    ▼
LayerNorm → LM Head (Linear → vocab logits)
    │
    ▼
Cross-Entropy Loss (training) / Sampling (inference)
```

## Files

| File | What it teaches |
|------|----------------|
| `model.py` | Full Transformer: attention, FFN, residuals, weight tying |
| `config.py` | Hyperparameter presets (nano / small) |
| `tokenizer.py` | Character-level byte tokenizer |
| `dataset.py` | Sliding-window next-token prediction dataset |
| `train.py` | Training loop, LR schedule, checkpointing, generation samples |
| `generate.py` | Load a checkpoint and generate text |

## Quick Start

```bash
# 1. Create and activate environment
conda create -n minillm_env python=3.11 -y
conda activate minillm_env
pip install -r requirements.txt

# 2. Train on the built-in tiny corpus (runs on CPU, ~1 min)
python train.py

# 3. Generate text from the trained model
python generate.py --checkpoint checkpoints/best.pt --prompt "The transformer"

# 4. Train on your own text file
python train.py --data your_text.txt --preset small
```

## Key Concepts Illustrated

### Causal Self-Attention
```
score(Q, K) = softmax(Q @ K.T / sqrt(d_k))
output      = score @ V
```
The causal mask ensures position `t` can only attend to positions `≤ t`.

### Why Residual Connections?
`x = x + sublayer(LayerNorm(x))`
Residuals create a "highway" for gradients — without them deep networks are
very hard to train.

### Weight Tying
The token embedding matrix and the LM head share the same weights.
Tokens that appear in similar contexts get similar embeddings, and the model
can directly compare hidden states to embeddings when predicting.

### Temperature & Top-k Sampling
- **temperature = 1.0**: sample from the raw distribution
- **temperature < 1.0**: sharper distribution (more repetitive but coherent)
- **temperature > 1.0**: flatter distribution (more creative but risky)
- **top_k = 40**: only sample from the 40 most likely tokens

## Model Sizes

| Preset | Params | n_embd | n_heads | n_layers | block_size |
|--------|--------|--------|---------|----------|------------|
| nano   | ~50K   | 64     | 4       | 4        | 128        |
| small  | ~1.5M  | 256    | 8       | 6        | 256        |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original Transformer paper
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for this project
