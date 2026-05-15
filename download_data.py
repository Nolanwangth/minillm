"""
Download small training datasets for MiniLLM.

Usage:
    python download_data.py shakespeare      # ~1MB, classic char-level benchmark
    python download_data.py tinystories      # ~2MB subset of simple English stories
    python download_data.py all              # download everything

Downloaded files are saved to the data/ directory.
Pass the file path to train.py with --data data/<file>.txt
"""

import argparse
import os
import urllib.request


DATA_DIR = "data"

DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "shakespeare.txt",
        "desc": "TinyShakespeare (~1MB) — all Shakespeare works concatenated",
    },
    "tinystories": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "filename": "tinystories_valid.txt",
        "desc": "TinyStories validation split (~2MB) — simple English stories for small LMs",
    },
}


def download(name: str):
    info = DATASETS[name]
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, info["filename"])

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"[{name}] Already exists at {dest} ({size:,} bytes). Skipping.")
        return dest

    print(f"[{name}] {info['desc']}")
    print(f"[{name}] Downloading from {info['url']} ...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:.1f}%  ({downloaded:,} / {total_size:,} bytes)", end="", flush=True)

    urllib.request.urlretrieve(info["url"], dest, reporthook=progress)
    print(f"\n[{name}] Saved to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        path = download(name)
        size = os.path.getsize(path)
        print(f"  -> {path}  ({size:,} bytes)\n")

    print("Done! Train with:")
    for name in targets:
        fname = DATASETS[name]["filename"]
        print(f"  python train.py --data data/{fname}")


if __name__ == "__main__":
    main()
