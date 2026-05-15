"""
MiniLLM — A tiny but complete GPT-style language model built from scratch.

Architecture (decoder-only Transformer):
  Token Embedding → Positional Encoding → N × TransformerBlock → LayerNorm → LM Head

Each TransformerBlock:
  LayerNorm → CausalMultiHeadAttention → residual
  LayerNorm → FeedForward → residual
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal (autoregressive) mask.

    Each token can only attend to itself and previous tokens — this is what
    makes the model generative: it can't "cheat" by looking at future tokens.

    Q, K, V are all projected from the same input x (self-attention).
    We split the embedding into `n_heads` independent heads, run attention
    in parallel, then concatenate and project back.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        # Single matrix for Q, K, V projections (3x for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of ones
        # Registered as a buffer so it moves with the model to GPU/CPU
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project to Q, K, V and split into heads
        q, k, v = self.c_attn(x).split(C, dim=2)
        # Reshape: (B, T, C) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        # Apply causal mask: set future positions to -inf so they vanish after softmax
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # reassemble heads
        return self.resid_drop(self.c_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN).

    Applied independently to each position. Expands the embedding to 4x width,
    applies a nonlinearity (GELU), then projects back. This is where most of
    the model's "memory" capacity lives.
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer layer = attention + FFN, each with pre-LayerNorm and residual.

    Pre-norm (LayerNorm before attention/FFN) stabilizes training compared to
    the original post-norm design in "Attention Is All You Need".
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # attention sub-layer with residual
        x = x + self.ff(self.ln2(x))    # FFN sub-layer with residual
        return x


class MiniLLM(nn.Module):
    """
    Decoder-only GPT-style language model.

    Given a sequence of token IDs, predicts the next token at every position.
    During training we minimize cross-entropy loss over all positions.
    During inference we sample autoregressively token by token.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "tok_emb": nn.Embedding(config.vocab_size, config.n_embd),
                "pos_emb": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "blocks": nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        # LM head: maps final hidden state back to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head.
        # This is a classic trick that reduces parameters and helps generalisation.
        self.transformer["tok_emb"].weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token IDs
        targets: (B, T) next-token labels (same as idx shifted by 1)
        returns: logits (B, T, vocab_size), and loss if targets provided
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence longer than block_size"

        # Token + positional embeddings
        tok = self.transformer["tok_emb"](idx)
        pos = self.transformer["pos_emb"](torch.arange(T, device=idx.device))
        x = self.transformer["drop"](tok + pos)

        # Pass through all Transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation: feed the model its own output one token at a time.

        temperature: >1 = more random, <1 = more greedy
        top_k:       keep only the top-k most likely tokens before sampling
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last position only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
