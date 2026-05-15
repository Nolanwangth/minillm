"""
Character-level byte tokenizer.

Simplest possible tokenizer: each UTF-8 byte is one token.
No special tokens, no vocabulary file needed — vocab_size is always 256.

For a real LLM you'd use BPE (tiktoken / sentencepiece), but for learning
purposes this is perfect because it's completely transparent.
"""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")
