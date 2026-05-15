"""
字节级分词器（Tokenizer）。

分词器的作用：把人类读的文字，转换成模型能处理的数字序列。

这里用最简单的方案：每个 UTF-8 字节就是一个 token。
- 英文字母：每个字母 = 1 个字节 = 1 个 token
- 中文字符：每个汉字 = 3 个字节 = 3 个 token
- 词表大小永远是 256（一个字节最多 256 种取值）

真实的 LLM（GPT-4、LLaMA）用 BPE（字节对编码）分词，
会把常见的字母组合合并成一个 token（比如 "ing"、"tion"），
词表通常有 3 万～10 万个 token，效率更高。
但对于学习来说，字节级最透明，没有任何魔法。
"""


class ByteTokenizer:
    vocab_size = 256  # 一个字节 = 8 位 = 256 种可能值

    def encode(self, text: str) -> list[int]:
        """把字符串转成整数列表。"""
        # str.encode("utf-8") 把字符串编成字节序列
        # list() 把每个字节转成 0-255 的整数
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """把整数列表转回字符串。"""
        # bytes(ids) 把整数列表变成字节序列
        # errors="replace" 表示遇到非法字节就用 ? 替代，不会崩溃
        return bytes(ids).decode("utf-8", errors="replace")
