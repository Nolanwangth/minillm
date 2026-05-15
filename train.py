"""
Training script for MiniLLM.

Usage:
    python train.py                        # train on built-in tiny corpus
    python train.py --data myfile.txt      # train on your own text file

What this script teaches:
  1. How to set up a training loop for a language model
  2. Gradient accumulation (simulate larger batch sizes)
  3. Learning rate warmup + cosine decay
  4. Periodic evaluation and checkpoint saving
  5. Autoregressive generation to see what the model learned
"""

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from config import NANO, SMALL, ModelConfig
from dataset import TextDataset, load_text
from model import MiniLLM
from tokenizer import ByteTokenizer

# ── Built-in tiny training corpus ────────────────────────────────────────────
TINY_CORPUS = """
The transformer architecture is the foundation of modern large language models.
It was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.

A transformer consists of stacked layers, each containing two sub-layers:
multi-head self-attention and a position-wise feed-forward network.

Self-attention allows each token to look at every other token in the sequence
and compute a weighted combination based on relevance (query-key similarity).
This is what gives transformers their power: long-range dependencies are easy.

The key equation is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Multi-head attention runs this in parallel with different learned projections,
allowing the model to attend to information from different representation subspaces.

Positional encodings are added to token embeddings because attention itself
is permutation-invariant — it has no notion of order without them.

The feed-forward network (FFN) is applied independently at each position.
It expands the hidden dimension by 4x, applies a nonlinearity, then projects back.
Most of the model's "factual knowledge" is thought to be stored in these weights.

Layer normalization and residual connections are used throughout to stabilise training.
Pre-LayerNorm (applying norm before each sub-layer) works better than the original post-norm.

GPT-style models (decoder-only) use causal masking so each token can only
attend to itself and previous tokens, making autoregressive generation possible.

Training a language model means minimising cross-entropy loss:
  predict the next token given all previous tokens.
  loss = -log P(x_{t+1} | x_1, ..., x_t)

Inference is done autoregressively: generate one token at a time, append it
to the context, and feed the extended context back into the model.

Temperature controls the sharpness of the output distribution.
Top-k sampling restricts choices to the k most probable tokens.
""".strip()


# ── Learning rate schedule ─────────────────────────────────────────────────────
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup + cosine decay — the standard LLM schedule."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Evaluation helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--preset", choices=["nano", "small"], default="nano")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--generate_every", type=int, default=20, help="Generate sample every N epochs")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    text = load_text(args.data) if args.data else TINY_CORPUS
    print(f"Corpus size: {len(text):,} characters")

    tokenizer = ByteTokenizer()
    config = NANO if args.preset == "nano" else SMALL
    config.vocab_size = tokenizer.vocab_size

    dataset = TextDataset(text, tokenizer, config.block_size)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MiniLLM(config).to(device)
    print(f"Model parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = total_steps // 10

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(step, warmup_steps, total_steps, args.lr, args.lr / 10)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            # Gradient clipping: prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()
            step += 1

        epoch_loss /= len(train_loader)
        val_loss = estimate_loss(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train loss: {epoch_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"lr: {lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "config": config, "val_loss": val_loss},
                os.path.join(args.save_dir, "best.pt"),
            )

        # Periodic generation to see progress
        if epoch % args.generate_every == 0:
            prompt = "The transformer"
            model.eval()
            encoded = torch.tensor([tokenizer.encode(prompt)], device=device)
            out = model.generate(encoded, max_new_tokens=100, temperature=0.8, top_k=40)
            generated = tokenizer.decode(out[0].tolist())
            print(f"\n--- Sample (epoch {epoch}) ---\n{generated}\n{'---' * 10}\n")
            model.train()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {args.save_dir}/best.pt")


if __name__ == "__main__":
    main()
