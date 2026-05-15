# 从零学 LLM — 一步一步指南

按这个顺序走，每个阶段都告诉你：读哪个文件、在 Cursor 里打哪里的断点、关注什么。

**调试方式：** 用 Cursor 打开 `minillm` 文件夹，打红点断点，按 F5 选对应配置。

---

## 准备工作

```bash
conda activate minillm_env
cd /Users/nolan/Desktop/agi/minillm
```

---

## 第一阶段：数据是什么？（10 分钟）

**目标：** 理解 LLM 吃的是什么形式的数据。

**读：** `tokenizer.py`

**关注：**
- `encode()` 把字符串变成数字列表
- `decode()` 把数字列表变回字符串
- vocab_size = 256，每个字节是一个 token

**断点位置：** `tokenizer.py` 的 `encode` 方法第一行，F5 选"训练 (nano, 内置语料)"。
停下来后看 `text` 变量是什么，单步 F10 看 `list(text.encode("utf-8"))` 返回了什么数字。

**核心概念：** LLM 看到的不是文字，是数字序列。

---

## 第二阶段：训练数据长什么样？（15 分钟）

**目标：** 理解"预测下一个 token"这个训练目标。

**读：** `dataset.py`，重点看 `__getitem__`

**关注：**
- `x` 和 `y` 的关系：y 就是 x 整体向右移一位
- next-token prediction：给你前面的，预测下一个

**断点位置：** `dataset.py` 的 `__getitem__` 里 `x = chunk[:-1]` 这行。
停下来后展开 `chunk`，看它是一段连续的 token ID。
单步走，对比 `x` 和 `y` 的第一个值，确认 y[0] = x[1]。

**核心概念：** 训练就是反复问"下一个字是什么"，答错了调参数。

---

## 第三阶段：模型配置（5 分钟）

**读：** `config.py`

**关注这 4 个参数：**

| 参数 | 含义 | 类比 |
|------|------|------|
| `n_embd` | 每个 token 用多少维向量表示 | 描述精细度 |
| `n_heads` | 注意力头数 | 同时从几个角度看 |
| `n_layers` | Transformer 层数 | 思考深度 |
| `block_size` | 最长记住多少 token | 记忆长度 |

不需要断点，直接读注释就够了。

---

## 第四阶段：注意力机制（30 分钟）⭐ 最重要

**目标：** 整个项目的核心，花时间搞懂。

**读：** `model.py`，先只看 `CausalSelfAttention.forward`

**断点打这几行，逐个观察形状变化：**

```python
q, k, v = self.c_attn(x).split(C, dim=2)
# 停在这里：展开 q，看它的 shape 是 (B, T, C)

q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
# 停在这里：看 q 的 shape 变成了 (B, n_heads, T, head_dim)

attn = (q @ k.transpose(-2, -1)) * scale
# 停在这里：看 attn 的 shape 是 (B, n_heads, T, T)
# T×T 矩阵：每个 token 对每个 token 的注意力分数

attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
# 停在这里：看 mask 长什么样，右上角全是 -inf
```

**Q/K/V 是什么：**
- Q（Query）：我想找什么
- K（Key）：我有什么
- V（Value）：实际的内容
- 点积 QK^T 衡量相关性，再用来加权 V

**核心概念：** "it was tired" 里的 "it" 指向 "animal"，靠的就是注意力。

---

## 第五阶段：完整模型（20 分钟）

**读：** `model.py` 全部

**断点打 `MiniLLM.forward` 里：**

```python
tok = self.transformer["tok_emb"](idx)
# 停这里：idx 是整数，tok 是向量，shape 从 (B,T) → (B,T,64)

x = self.transformer["drop"](tok + pos)
# 停这里：tok + pos 就是"词义 + 位置"的叠加

for block in self.transformer["blocks"]:
    x = block(x)
# 在这行打断点，按 F10 走 4 次（4 层），每次看 x 的 shape——始终是 (B,T,64)
# 形状不变，但内容在变，这就是 Transformer 的工作方式

logits = self.lm_head(x)
# 停这里：shape 从 (B,T,64) → (B,T,256)，256 是词表大小
```

**关注残差连接（在 `TransformerBlock.forward`）：**
```python
x = x + self.attn(self.ln1(x))   # 这个 x + 非常关键
x = x + self.ff(self.ln2(x))
```
去掉这两个 `+` 试试，loss 会很难下降。

---

## 第六阶段：跑第一次训练（20 分钟）

F5 选"训练 (nano, 内置语料)"，不打断点，看终端输出：

```
Epoch   1/100 | train loss: 5.5123 | val loss: 5.4987 | lr: 3.00e-05 | 0.3s
Epoch  50/100 | train loss: 3.1234 | val loss: 3.2011 | lr: 1.80e-04 | 0.3s
```

**关注：**
- loss 从 ~5.5 往下降 → 模型在学习
- train loss 和 val loss 差不多 → 没有过拟合
- lr 先从小变大（warmup），再慢慢变小（cosine decay）

**断点打 `train.py` 的 `get_lr` 函数**，传不同的 `step` 值进去，
看学习率曲线是怎么算出来的。

---

## 第七阶段：莎士比亚训练（30 分钟）

F5 选"训练 (Shakespeare)"，等它跑完，观察每 5 个 epoch 的生成样本从乱码变成有点像莎士比亚的句子。

**断点打 `train.py` 里的生成代码：**
```python
out = model.generate(encoded, max_new_tokens=100, temperature=0.8, top_k=40)
```
进入 `model.generate`，断点打在循环里，单步走看每次是怎么生成一个 token 的。

---

## 第八阶段：推理和采样（15 分钟）

F5 选"推理生成"，在 `model.generate` 里打断点。

**重点观察：**

```python
logits = logits[:, -1, :] / temperature
# 停这里：只取最后一个位置的 logits，temperature 控制尖锐程度

v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = float("-inf")
# 停这里：top_k 过滤，看过滤前后 logits 的变化

next_token = torch.multinomial(probs, num_samples=1)
# 停这里：看最终采样到的 token 是什么数字，decode 回去是什么字
```

---

## 第九阶段：SFT 微调（20 分钟）⭐ 进阶

**目标：** 理解预训练和微调的核心区别。

**先读：** `sft_data.py`，重点看 `SFTDataset.__init__`

**断点打这里：**
```python
labels[response_start:] = input_ids[response_start:]
```
停下来后对比 `labels` 和 `input_ids`：
- `labels` 中 user 部分全是 `-100`（屏蔽）
- `labels` 中 assistant 部分和 `input_ids` 一样（计算 loss）

**然后打开 `sft.py`，断点打这行：**
```python
loss = torch.nn.functional.cross_entropy(..., ignore_index=-100)
```
这一行就是 SFT 和预训练的唯一本质区别。

**跑 SFT：** F5 选"SFT 微调"（需要先有 `checkpoints/best.pt`）

---

## 第十阶段：做实验（随意探索）

### 实验 1：去掉残差
`TransformerBlock.forward` 里把 `x = x + ...` 改成 `x = ...`，重训，看 loss 还能不能收敛。

### 实验 2：去掉因果掩码
`CausalSelfAttention.forward` 里注释掉 `masked_fill` 那行，看生成时会怎样。

### 实验 3：改模型大小
`config.py` 里把 `n_layers=4` 改成 `n_layers=1`，重训，对比生成质量。

### 实验 4：调采样参数
推理时 `--temp 0.1` vs `--temp 2.0`，`--top_k 1` vs `--top_k 200`，感受差异。

### 实验 5：加自己的 SFT 数据
写一个 `mydata.jsonl`，每行 `{"user": "...", "assistant": "..."}`，
然后 `python sft.py --checkpoint checkpoints/best.pt --data mydata.jsonl`

---

## 学完之后你掌握了什么

```
预训练部分：
✅ Tokenization（文字 → 数字）
✅ Token / Positional Embedding（数字 → 向量，加位置信息）
✅ Causal Self-Attention（token 互相看，但不看未来）
✅ Multi-Head Attention（多角度注意力）
✅ Feed-Forward Network（每个位置独立处理）
✅ Residual Connection + LayerNorm（训练稳定的关键）
✅ Next-Token Prediction + Cross-Entropy Loss
✅ LR Warmup + Cosine Decay（训练技巧）
✅ Temperature & Top-k Sampling（推理技巧）

SFT 部分：
✅ 对话数据格式（[USER] / [ASST] / [END]）
✅ Loss Mask（只对回复部分算 loss）
✅ 灾难性遗忘（为什么 SFT 学习率要更小）
```

这就是 GPT-2、GPT-3、LLaMA 的核心。规模更大，原理完全一样。

---

## 下一步

1. **[nanoGPT](https://github.com/karpathy/nanoGPT)** — Karpathy 的版本，真正在 GPT-2 规模上训练
2. **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)** — 原始论文逐行注释
3. **[LLaMA 3 论文](https://arxiv.org/abs/2407.21783)** — RoPE、GQA、SwiGLU 等工业级改进
