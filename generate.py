"""
推理脚本：从训练好的 checkpoint 加载模型并生成文字。

用法：
    python generate.py --checkpoint checkpoints/best.pt --prompt "The transformer"
    python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --tokens 200 --temp 0.7

参数说明：
    --checkpoint  训练保存的 .pt 文件路径
    --prompt      给模型的起始文字（模型会续写这段文字）
    --tokens      最多生成多少个 token
    --temp        温度。越低越保守，越高越随机
    --top_k       只从概率最高的 k 个 token 里采样
"""

import argparse

import torch

from model import MiniLLM
from tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The transformer")
    parser.add_argument("--tokens", type=int, default=150)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    # 自动选择最快的设备：CUDA GPU > Apple MPS > CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ── 加载 checkpoint ────────────────────────────────────────────────────────
    # checkpoint 是一个字典，包含：模型权重、配置、训练到第几个 epoch、val loss
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    # 用保存的配置重建模型结构，再加载权重
    model = MiniLLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # 切换到推理模式（关闭 Dropout）

    print(f"加载 checkpoint：epoch {ckpt['epoch']}，val loss: {ckpt['val_loss']:.4f}")
    print(f"模型参数量：{model.num_params():,}\n")

    # ── 编码 prompt ────────────────────────────────────────────────────────────
    tokenizer = ByteTokenizer()
    # 把 prompt 文字编码成 token ID，包成 batch 维度（加一个维度变成 (1, T)）
    encoded = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    print(f"输入 prompt：{args.prompt!r}")
    print("=" * 60)

    # ── 生成 ───────────────────────────────────────────────────────────────────
    # torch.no_grad()：推理时不需要计算梯度，关掉可以节省内存和加速
    with torch.no_grad():
        out = model.generate(
            encoded,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
        )

    # out 是 (1, T+生成数) 的 tensor，取第 0 个样本，转成列表再解码
    generated = tokenizer.decode(out[0].tolist())
    print(generated)
    print("=" * 60)


if __name__ == "__main__":
    main()
