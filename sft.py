"""
SFT（监督微调）训练脚本。

流程：
  1. 加载预训练 checkpoint（train.py 产出的 best.pt）
  2. 在问答数据上继续训练
  3. 只对 assistant 回复部分计算 loss
  4. 保存微调后的 checkpoint

用法：
  # 先跑预训练
  python train.py --epochs 200 --data data/shakespeare.txt

  # 再跑 SFT（加载预训练权重）
  python sft.py --checkpoint checkpoints/best.pt

  # 用自己的问答数据
  python sft.py --checkpoint checkpoints/best.pt --data mydata.jsonl

  # 用 SFT 后的模型生成
  python generate.py --checkpoint checkpoints/sft_best.pt --prompt "[USER]什么是注意力机制？[ASST]"

JSONL 数据格式（每行一条）：
  {"user": "你的问题", "assistant": "模型的回答"}
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, random_split

from model import MiniLLM
from sft_data import BUILTIN_DATA, SFTDataset, format_conversation, load_jsonl
from tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="预训练 checkpoint 路径")
    parser.add_argument("--data", type=str, default=None, help="JSONL 问答数据路径（不填则用内置数据）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="SFT 学习率比预训练小 3-10 倍，防止灾难性遗忘")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ── 加载预训练模型 ────────────────────────────────────────────────────────
    # SFT 从预训练权重出发，不是从随机初始化开始
    # 这样模型已经有了语言能力，SFT 只是教它怎么用这个能力回答问题
    print(f"加载预训练 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    model = MiniLLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"预训练 epoch: {ckpt['epoch']}，val loss: {ckpt['val_loss']:.4f}")
    print(f"模型参数量: {model.num_params():,}\n")

    # ── 准备数据 ──────────────────────────────────────────────────────────────
    tokenizer = ByteTokenizer()
    raw_data = load_jsonl(args.data) if args.data else BUILTIN_DATA
    print(f"问答条数: {len(raw_data)}")

    # 打印几条样本，让你看清楚训练数据长什么样
    print("\n── 数据样例 ──────────────────────────────────────")
    for item in raw_data[:2]:
        print(format_conversation(item["user"], item["assistant"]))
        print()
    print("─" * 50 + "\n")

    dataset = SFTDataset(raw_data, tokenizer, config.block_size)

    # 数据少时不做验证集分割
    if len(dataset) >= 4:
        val_size = max(1, len(dataset) // 5)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds = val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    # SFT 学习率要比预训练小，防止"灾难性遗忘"：
    # 学习率太大会把预训练学到的知识覆盖掉
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)  # 不传 targets，因为 y 里有 IGNORE_INDEX 需要特殊处理

            # 手动计算 loss，ignore_index=-100 会自动跳过 user 部分
            # 这是 SFT 和预训练最核心的区别：只对 assistant 回复算 loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
                )
                val_losses.append(val_loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train loss: {epoch_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        # 保存最好的 checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "config": config, "val_loss": val_loss},
                os.path.join(args.save_dir, "sft_best.pt"),
            )

        # 每 10 个 epoch 生成一条样本看效果
        if epoch % 10 == 0:
            model.eval()
            test_prompt = "[USER]什么是自注意力？[ASST]"
            encoded = torch.tensor([tokenizer.encode(test_prompt)], device=device)
            with torch.no_grad():
                out = model.generate(encoded, max_new_tokens=80, temperature=0.7, top_k=40)
            print(f"\n── 生成样例 (epoch {epoch}) ──")
            print(tokenizer.decode(out[0].tolist()))
            print("─" * 40 + "\n")
            model.train()

    print(f"\nSFT 完成。最优 val loss: {best_val_loss:.4f}")
    print(f"Checkpoint 保存到 {args.save_dir}/sft_best.pt")
    print("\n推理命令：")
    print(f'  python generate.py --checkpoint {args.save_dir}/sft_best.pt --prompt "[USER]什么是 Transformer？[ASST]"')


if __name__ == "__main__":
    main()
