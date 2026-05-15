"""
Inference / generation script for a trained MiniLLM checkpoint.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "The transformer"
    python generate.py --checkpoint checkpoints/best.pt --prompt "Attention is" --tokens 200 --temp 0.7
"""

import argparse

import torch

from model import MiniLLM
from tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The transformer")
    parser.add_argument("--tokens", type=int, default=150, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling cutoff")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model = MiniLLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val loss: {ckpt['val_loss']:.4f})")
    print(f"Model params: {model.num_params():,}\n")

    tokenizer = ByteTokenizer()
    encoded = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    print(f"Prompt: {args.prompt!r}\n")
    print("=" * 60)

    with torch.no_grad():
        out = model.generate(encoded, max_new_tokens=args.tokens, temperature=args.temp, top_k=args.top_k)

    generated = tokenizer.decode(out[0].tolist())
    print(generated)
    print("=" * 60)


if __name__ == "__main__":
    main()
