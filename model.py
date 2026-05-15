"""
MiniLLM — 从零构建的 GPT 风格语言模型

整体架构（仅解码器的 Transformer）：
  词元嵌入 → 位置嵌入 → N 个 TransformerBlock → LayerNorm → 语言模型头

每个 TransformerBlock 包含两个子层：
  LayerNorm → 因果多头自注意力 → 残差连接
  LayerNorm → 前馈网络         → 残差连接
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    带因果掩码的多头自注意力层。

    "因果"的意思：每个位置只能看到它自己和它左边的 token，
    不能看右边的未来 token。这正是语言模型能自回归生成的原因——
    预测第 t 个 token 时，不能作弊去看第 t+1、t+2……

    Q/K/V 都从同一个输入 x 投影而来，所以叫"自"注意力。
    把 embedding 拆成 n_heads 个独立的头并行计算，
    最后拼接起来再投影回原来的维度。
    """

    def __init__(self, config):
        super().__init__()
        # embedding 维度必须能被头数整除，才能均匀拆分
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads  # 每个头的维度

        # 用一个大矩阵同时投影出 Q、K、V，比三个矩阵分开写更高效
        # 输入: (B, T, n_embd)  输出: (B, T, 3*n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # 把多头的输出拼接后再投影回 n_embd 维
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)   # 注意力权重上的 dropout
        self.resid_drop = nn.Dropout(config.dropout)  # 输出上的 dropout

        # 因果掩码：下三角矩阵（对角线及左下角为 1，右上角为 0）
        # 形状: (1, 1, block_size, block_size)，前两个维度是 batch 和 head 的广播维度
        # register_buffer 的好处：不是可训练参数，但会随模型一起移动到 GPU/CPU
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape  # B=batch大小, T=序列长度, C=embedding维度(n_embd)

        # ── 第一步：投影出 Q、K、V ────────────────────────────────────────────
        # c_attn 输出 (B, T, 3*C)，split 按最后一维切成三份，各 (B, T, C)
        q, k, v = self.c_attn(x).split(C, dim=2)

        # ── 第二步：拆分多头 ──────────────────────────────────────────────────
        # (B, T, C) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        # 转置是为了让注意力计算在 (T, head_dim) 这两个维度上进行
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ── 第三步：缩放点积注意力 ────────────────────────────────────────────
        # 公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        #
        # QK^T: (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T)
        #     = (B, n_heads, T, T)  ← 每对 token 之间的相似度得分
        #
        # 除以 sqrt(head_dim) 是为了防止点积值太大导致 softmax 梯度消失
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # ── 第四步：应用因果掩码 ──────────────────────────────────────────────
        # mask 中为 0 的位置（右上角，代表"未来"）填成 -inf
        # softmax(-inf) = 0，所以这些位置的注意力权重变为 0，相当于被屏蔽
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)  # 在最后一维（key 方向）做 softmax
        attn = self.attn_drop(attn)

        # ── 第五步：用注意力权重对 V 做加权求和 ──────────────────────────────
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) = (B, n_heads, T, head_dim)
        out = attn @ v

        # ── 第六步：合并多头，投影回原维度 ────────────────────────────────────
        # (B, n_heads, T, head_dim) → (B, T, n_heads, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class FeedForward(nn.Module):
    """
    逐位置前馈网络（FFN）。

    对序列中每个位置独立做两层线性变换，中间用 GELU 激活。
    先把维度扩大 4 倍再压缩回来，这个"先扩后缩"的结构
    让模型有更大的空间来存储和变换信息。

    GPT-3 的大部分参数都在 FFN 层里，
    研究者认为模型的"事实知识"主要存在这里。
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层：扩大 4 倍  (B, T, n_embd) → (B, T, 4*n_embd)
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            # GELU 激活：比 ReLU 更平滑，现代 LLM 的标准选择
            nn.GELU(),
            # 第二层：压缩回来  (B, T, 4*n_embd) → (B, T, n_embd)
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 层 = 注意力 + FFN，各带 Pre-LayerNorm 和残差连接。

    Pre-LayerNorm（先 Norm 再计算）比原始论文的 Post-LayerNorm 训练更稳定，
    现代所有主流 LLM（GPT-2 之后）都用 Pre-LayerNorm。

    残差连接（x = x + sublayer(x)）的作用：
    - 让梯度可以直接从深层流回浅层，解决梯度消失
    - 相当于给每一层加了一条"高速公路"，信息可以绕过这层直接传递
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)      # 注意力前的 LayerNorm
        self.attn = CausalSelfAttention(config)      # 因果自注意力
        self.ln2 = nn.LayerNorm(config.n_embd)      # FFN 前的 LayerNorm
        self.ff = FeedForward(config)                # 前馈网络

    def forward(self, x):
        # 注意力子层：先 Norm，做注意力，加回原始 x（残差）
        x = x + self.attn(self.ln1(x))
        # FFN 子层：先 Norm，过 FFN，加回原始 x（残差）
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """
    仅解码器的 GPT 风格语言模型。

    训练时：给定 token 序列，预测每个位置的下一个 token，用交叉熵 loss 训练。
    推理时：给定一个 prompt，逐个 token 自回归地生成后续内容。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            # Token 嵌入：把 token ID（整数）映射成 n_embd 维的向量
            # 词表大小 × embedding 维度的查找表
            "tok_emb": nn.Embedding(config.vocab_size, config.n_embd),

            # 位置嵌入：把位置索引（0,1,2,...）映射成向量
            # 因为注意力本身不感知顺序，必须显式告诉模型"这是第几个 token"
            "pos_emb": nn.Embedding(config.block_size, config.n_embd),

            "drop": nn.Dropout(config.dropout),

            # 堆叠 n_layers 个 TransformerBlock
            "blocks": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),

            # 最后一个 LayerNorm，在输出 logits 之前做归一化
            "ln_f": nn.LayerNorm(config.n_embd),
        })

        # 语言模型头：把 n_embd 维的隐藏状态映射回词表大小，得到每个 token 的得分（logits）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享（Weight Tying）：让 tok_emb 和 lm_head 共用同一组参数
        # 直觉：在相似语境中出现的 token 会有相似的 embedding，
        # 而 lm_head 预测时也在比较隐藏状态和 token embedding 的相似度，
        # 两者共享权重更合理，同时还能减少参数量
        self.transformer["tok_emb"].weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        # 用均值 0、标准差 0.02 的正态分布初始化所有权重
        # 这个初始化方式来自 GPT-2，经验上比默认初始化训练更稳定
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播。

        参数：
            idx:     (B, T) 整数张量，token ID 序列
            targets: (B, T) 整数张量，下一个 token 的标签（训练时用）

        返回：
            logits: (B, T, vocab_size) 每个位置对应词表中每个 token 的得分
            loss:   标量，交叉熵损失（只在传入 targets 时计算）
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大上下文长度 {self.config.block_size}"

        # ── 嵌入层 ────────────────────────────────────────────────────────────
        tok = self.transformer["tok_emb"](idx)   # (B, T, n_embd)  词元嵌入
        # arange(T) = [0, 1, 2, ..., T-1]，位置索引
        pos = self.transformer["pos_emb"](torch.arange(T, device=idx.device))  # (T, n_embd)
        # tok + pos：广播相加，给每个 token 的向量加上它的位置信息
        x = self.transformer["drop"](tok + pos)  # (B, T, n_embd)

        # ── 逐层通过 Transformer Block ────────────────────────────────────────
        for block in self.transformer["blocks"]:
            x = block(x)  # 形状保持 (B, T, n_embd) 不变

        # ── 输出层 ────────────────────────────────────────────────────────────
        x = self.transformer["ln_f"](x)          # 最终 LayerNorm
        logits = self.lm_head(x)                 # (B, T, vocab_size)

        # ── 计算损失（训练时） ────────────────────────────────────────────────
        loss = None
        if targets is not None:
            # 把 (B, T, vocab_size) 展平成 (B*T, vocab_size)
            # 把 (B, T) 展平成 (B*T,)
            # 交叉熵 = -log(正确 token 的概率)，对所有位置取平均
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归文本生成。

        每次只生成一个 token，然后把它追加到输入序列末尾，
        再把新序列喂回模型，循环 max_new_tokens 次。

        参数：
            temperature: 控制随机程度。
                         = 1.0：按原始概率分布采样
                         < 1.0：概率分布更尖锐，输出更保守/重复
                         > 1.0：概率分布更平坦，输出更随机/有创意
            top_k:       只从概率最高的 k 个 token 里采样，过滤掉低概率的噪声
        """
        for _ in range(max_new_tokens):
            # 如果上下文超过 block_size，截取最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            # 只取最后一个位置的 logits（我们要预测序列末尾的下一个 token）
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k 过滤：把不在 top-k 的位置设为 -inf
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # softmax 转成概率，然后按概率随机采样一个 token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 把新生成的 token 追加到序列末尾
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)

        return idx

    def num_params(self):
        """返回模型总参数量。"""
        return sum(p.numel() for p in self.parameters())
