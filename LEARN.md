# 从零学 LLM — 一步一步指南

这份指南告诉你：按什么顺序读代码、在终端输入什么、每一步重点关注什么。
**不需要任何 AI 背景，只需要会 Python。**

---

## 准备工作（只做一次）

```bash
conda activate minillm_env
cd /Users/nolan/Desktop/AGI/minillm
```

---

## 第一阶段：数据是什么？（10 分钟）

**目标：** 理解 LLM 吃的是什么形式的数据。

### 读这个文件
`tokenizer.py` — 只有 10 行，最简单的起点

**关注：**
- `encode()` 把字符串变成数字列表
- `decode()` 把数字列表变回字符串
- vocab_size = 256，意味着每个字节是一个 token

### 在终端里玩

```python
# 启动 Python 交互环境
python

>>> from tokenizer import ByteTokenizer
>>> tok = ByteTokenizer()

>>> tok.encode("hello")
# 你会看到：[104, 101, 108, 108, 111]
# 每个数字对应一个字母的 ASCII 码

>>> tok.decode([104, 101, 108, 108, 111])
# 你会看到：'hello'

>>> tok.encode("你好")
# 汉字会被编成多个字节（UTF-8）
```

**记住这个核心概念：** LLM 看到的不是文字，是数字序列。

---

## 第二阶段：训练数据长什么样？（15 分钟）

**目标：** 理解"预测下一个 token"这个训练目标。

### 读这个文件
`dataset.py` — 重点看 `__getitem__`

**关注：**
- `x` 和 `y` 的关系：y 就是 x 整体向右移一位
- 这就是 next-token prediction：给你前面的，预测下一个

### 在终端里玩

```python
python

>>> from tokenizer import ByteTokenizer
>>> from dataset import TextDataset
>>> tok = ByteTokenizer()
>>> ds = TextDataset("hello world", tok, block_size=5)

>>> x, y = ds[0]
>>> print(x)   # tensor([104, 101, 108, 108, 111])  → "hello"
>>> print(y)   # tensor([101, 108, 108, 111,  32])  → "ello "

# x[0]=104('h'), y[0]=101('e') — 模型要学：看到'h'，预测'e'
# x[1]=101('e'), y[1]=108('l') — 看到'he'，预测'l'
# 以此类推...
```

**记住这个核心概念：** 训练就是反复问模型"下一个字是什么"，答错了就调整参数。

---

## 第三阶段：理解模型配置（5 分钟）

### 读这个文件
`config.py`

**关注这 4 个最重要的超参数：**

| 参数 | 含义 | 类比 |
|------|------|------|
| `n_embd` | 每个 token 用多少维向量表示 | 词语的"描述精细度" |
| `n_heads` | 注意力头数 | 同时从几个角度理解句子 |
| `n_layers` | Transformer 层数 | 思考深度 |
| `block_size` | 最长能记住多少个 token | 记忆长度 |

### 在终端里玩

```python
python

>>> from config import NANO, SMALL
>>> print(NANO)
# 看看 nano 预设的参数

>>> print(SMALL)
# 对比 small，参数大了很多
```

---

## 第四阶段：模型的核心——注意力机制（30 分钟）

**目标：** 这是整个项目最重要的部分，花时间理解它。

### 读这个文件
`model.py` — 先只看 `CausalSelfAttention` 类

**逐行关注：**

1. **`self.c_attn`**：一个 Linear 层，把输入同时投影成 Q、K、V 三份
2. **Q、K、V 是什么：**
   - Q (Query)：我想找什么信息
   - K (Key)：我有什么信息
   - V (Value)：实际的信息内容
   - 类比图书馆：Q 是你的问题，K 是书的标题，V 是书的内容
3. **`attn = (q @ k.transpose(-2,-1)) * scale`**：计算每个 token 对其他 token 的"关注度"
4. **`masked_fill(..., float("-inf"))`**：因果掩码，强制只能看左边，不能看右边
5. **`out = attn @ v`**：用关注度加权平均所有 token 的信息

### 在终端里玩（可视化注意力）

```python
python

import torch
from config import NANO
from model import CausalSelfAttention

cfg = NANO
attn = CausalSelfAttention(cfg)

# 模拟一个 batch：2个句子，每句8个token，64维
x = torch.randn(2, 8, 64)
out = attn(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
# 形状不变：注意力层不改变数据的形状，只是让每个位置"看到"了其他位置的信息
```

**记住这个核心概念：** 注意力机制让"the animal didn't cross the street because **it** was too tired"中的模型知道 "it" 指的是 "animal"。

---

## 第五阶段：完整模型结构（20 分钟）

### 继续读 `model.py`，这次看全部

**按这个顺序理解：**

```
FeedForward      → 每个位置独立"思考"，扩展4倍再压缩回来
TransformerBlock → 注意力 + FFN + 残差连接，组合成一层
MiniLLM.forward  → 整个前向传播流程
MiniLLM.generate → 推理时怎么一个一个生成 token
```

**关注残差连接：**
```python
x = x + self.attn(self.ln1(x))  # 注意这个 x +
x = x + self.ff(self.ln2(x))    # 还有这个 x +
```
这两个 `+` 非常重要——没有它们，深层网络根本训练不起来。

### 在终端里玩

```python
python

import torch
from config import NANO
from model import MiniLLM
from tokenizer import ByteTokenizer

tok = ByteTokenizer()
cfg = NANO
cfg.vocab_size = tok.vocab_size
model = MiniLLM(cfg)

print(f"参数量: {model.num_params():,}")
# 大约 222,336 个参数 — GPT-3 有 1750 亿，我们的只有它的百万分之一

# 统计每一层的参数量
for name, param in model.named_parameters():
    print(f"{name:50s} {param.numel():>10,}")
```

---

## 第六阶段：跑一次训练（20 分钟）

**目标：** 亲眼看到 loss 下降。

```bash
# 先用内置语料快速跑，看看输出格式
python train.py --epochs 50

# 关注终端输出：
# Epoch   1/50 | train loss: 5.5123 | val loss: 5.4987 | lr: 3.00e-05 | 0.3s
# Epoch  10/50 | train loss: 4.2341 | val loss: 4.3012 | lr: 2.73e-04 | 0.3s
# ...
# loss 从 ~5.5 下降，说明模型在学习
```

**关注这几个数字：**
- `train loss`：训练集上的损失，应该持续下降
- `val loss`：验证集上的损失，衡量泛化能力
- 如果 train loss 很低但 val loss 很高 → 过拟合了
- `lr`：学习率，前几轮从小到大 warmup，之后慢慢降

---

## 第七阶段：用真实数据训练（30 分钟）

```bash
# 用莎士比亚数据集（已经下载好了）
python train.py --data data/shakespeare.txt --epochs 200 --generate_every 50
```

每 50 个 epoch 模型会自动生成一段文字，**观察它从乱码变成有点像莎士比亚风格的句子**，这个过程很直观地告诉你模型在学什么。

早期输出（epoch 50）：
```
The transformerQ�k&�U7...  ← 还是乱码
```

后期输出（epoch 200）：
```
The transformer of the king, and the world...  ← 开始有点意思了
```

---

## 第八阶段：推理和采样（15 分钟）

训练完成后，用 `generate.py` 来玩：

```bash
# 基础生成
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"

# 调低温度：更保守、更确定
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --temp 0.3

# 调高温度：更随机、更有创意（但可能胡说）
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --temp 1.5

# 增加 top_k：从更多候选里采样
python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --top_k 5
```

**关注温度对输出的影响。** 这直接对应 ChatGPT 里的"创造力"滑块。

---

## 第九阶段：做实验，改参数（随意探索）

现在你已经懂了整个流程，可以开始做实验：

### 实验 1：改模型大小
编辑 `config.py`，把 `n_layers` 从 4 改成 2，重新训练，观察质量变化。

### 实验 2：去掉残差连接
在 `model.py` 的 `TransformerBlock.forward` 里，把：
```python
x = x + self.attn(self.ln1(x))
```
改成：
```python
x = self.attn(self.ln1(x))
```
重新训练，观察 loss 还能不能下降。（剧透：很难）

### 实验 3：去掉因果掩码
在 `CausalSelfAttention.forward` 里，注释掉 `masked_fill` 那行，看看生成时会发生什么。

### 实验 4：改采样策略
在 `generate.py` 里试试 `top_k=1`（贪心解码，每次选概率最高的），和 `top_k=256`（完全随机），对比生成质量。

---

## 知识地图

学完这个项目后，你已经理解了：

```
✅ Tokenization（文字→数字）
✅ Token Embedding（数字→向量）
✅ Positional Encoding（给序列加位置信息）
✅ Self-Attention（token 之间互相"看"）
✅ Causal Mask（只看左边，不看右边）
✅ Multi-Head Attention（多角度注意力）
✅ Feed-Forward Network（每个位置独立处理）
✅ Residual Connection（残差，让梯度能流动）
✅ Layer Normalization（稳定训练）
✅ Next-Token Prediction（训练目标）
✅ Cross-Entropy Loss（怎么量化预测好坏）
✅ Temperature & Top-k Sampling（控制生成多样性）
```

这就是 GPT-2、GPT-3、LLaMA 的核心架构。规模更大，但原理完全一样。

---

## 下一步

掌握了这个之后，推荐的路径：

1. **[nanoGPT](https://github.com/karpathy/nanoGPT)** — Karpathy 的版本，训练在 GPT-2 规模上
2. **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** — 带注释的原始论文实现
3. **[LLaMA 3 论文](https://arxiv.org/abs/2407.21783)** — 工业级 LLM 和这里的区别（RoPE、GQA、SwiGLU 等）
