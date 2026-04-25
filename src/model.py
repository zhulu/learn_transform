from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """正弦位置编码。

    Transformer 没有 RNN/CNN 那种天然的顺序结构，所以 token embedding 本身
    不包含“第几个词”的信息。位置编码会给每个位置生成一个固定向量，
    再加到 token embedding 上，让模型知道词序。

    输入/输出形状都为：
    - x: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # pe 的形状是 [max_len, d_model]，每一行代表一个位置的位置向量。
        pe = torch.zeros(max_len, d_model)
        # position: [max_len, 1]，表示位置 0, 1, 2, ...
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 控制不同维度上的正弦/余弦频率。
        # 偶数维使用 sin，奇数维使用 cos，这是原始 Transformer 论文的做法。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer 表示 pe 不是可训练参数，但会跟随模型移动到 GPU/CPU。
        # unsqueeze 后形状变为 [1, max_len, d_model]，方便和 batch 广播相加。
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """给输入 embedding 加上对应长度的位置编码。"""
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力。

    注意力的核心公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    多头注意力会把 d_model 拆成 num_heads 份，让不同 head 学习不同关系。
    输入形状：
    - query: [batch_size, query_len, d_model]
    - key:   [batch_size, key_len, d_model]
    - value: [batch_size, key_len, d_model]
    - mask:  可广播到 [batch_size, num_heads, query_len, key_len]

    输出形状：
    - [batch_size, query_len, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Q/K/V 是从输入向量投影出来的三个视角：
        # Q 代表“我要找什么”，K 代表“我有什么索引”，V 代表“真正要聚合的信息”。
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 多个 head 拼回 d_model 后，再过一层线性映射融合信息。
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        # 线性投影后拆成多头：
        # [B, L, D] -> [B, H, L, head_dim]
        q = self._split_heads(self.q_proj(query), batch_size)
        k = self._split_heads(self.k_proj(key), batch_size)
        v = self._split_heads(self.v_proj(value), batch_size)

        # q @ k^T 得到每个 query token 对每个 key token 的相似度分数。
        # scores: [B, H, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask 为 False 的位置不允许注意力看到，填成极小值后 softmax 约等于 0。
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 用注意力权重对 value 做加权求和。
        context = torch.matmul(attn, v)
        # [B, H, L, head_dim] -> [B, L, H, head_dim] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, query.size(1), -1)
        return self.out_proj(context)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """把最后一维 d_model 切成多个 head。"""
        x = x.view(batch_size, x.size(1), self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    """逐位置前馈网络。

    这层对序列中的每个位置独立应用同一个 MLP：
    [d_model -> d_ff -> d_model]

    注意它不会混合不同 token 的信息；token 间信息混合由 attention 负责。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Transformer Encoder 的一层。

    每层结构：
    1. self-attention：源句内部 token 互相看。
    2. 残差连接 + LayerNorm。
    3. feed-forward：对每个位置做非线性变换。
    4. 残差连接 + LayerNorm。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # encoder self-attention 中 query/key/value 都来自源句自身。
        attn = self.self_attn(x, x, x, src_mask)
        # 残差连接保留原始信息，LayerNorm 稳定训练。
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        return self.norm2(x + self.dropout(ff))


class DecoderLayer(nn.Module):
    """Transformer Decoder 的一层。

    每层结构：
    1. masked self-attention：目标句只能看当前位置及之前的 token。
    2. cross-attention：目标句 token 去看 encoder 输出的源句表示。
    3. feed-forward。

    decoder 比 encoder 多了 cross-attention，这正是翻译时对齐源句信息的地方。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # masked self-attention：防止训练时偷看未来目标词。
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        # cross-attention：query 来自 decoder，key/value 来自 encoder memory。
        cross_attn = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        ff = self.ff(x)
        return self.norm3(x + self.dropout(ff))


class Transformer(nn.Module):
    """完整的 Encoder-Decoder Transformer。

    训练时输入：
    - src:    [batch_size, src_len]，源语言 token id。
    - tgt_in: [batch_size, tgt_len - 1]，目标语言右移后的输入。

    输出：
    - logits: [batch_size, tgt_len - 1, tgt_vocab_size]

    训练标签是 tgt_out，也就是目标句去掉第一个 <bos> 后的序列。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.d_model = d_model
        # embedding 把 token id 映射到 d_model 维向量。
        # padding_idx 会让 <pad> 对应的 embedding 在训练中保持为 0。
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_id)
        self.positional = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        # ModuleList 用于堆叠多层 encoder/decoder，同时让 PyTorch 正确注册参数。
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)]
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._reset_parameters()

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """训练阶段前向传播。

        src 先经过 encoder 得到 memory；tgt_in 再通过 decoder 对 memory 做 cross-attention；
        最后 generator 把 decoder 隐状态映射成词表分类 logits。
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_in)
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt_in, memory, tgt_mask, src_mask)
        return self.generator(decoded)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """只运行 encoder，推理时也会单独调用。

        返回 memory，形状为 [batch_size, src_len, d_model]。
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        # 乘 sqrt(d_model) 是原始论文中的缩放，让 embedding 尺度与位置编码更匹配。
        x = self.dropout(self.positional(self.src_embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """只运行 decoder。

        Args:
            tgt: 当前已有的目标端 token id。
            memory: encoder 输出。
            tgt_mask: 目标端 padding + causal mask。
            src_mask: 源端 padding mask，用于 cross-attention。
        """
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        x = self.dropout(self.positional(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        if src_mask is None:
            # 如果没有传入 src_mask，默认所有源 token 都可见。训练路径会显式传入 mask。
            src_mask = torch.ones(
                tgt.size(0),
                1,
                1,
                memory.size(1),
                dtype=torch.bool,
                device=tgt.device,
            )
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return x

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """构造源端 padding mask。

        输入 src: [B, src_len]
        输出 mask: [B, 1, 1, src_len]

        这个形状可以广播到 attention scores 的 [B, H, query_len, src_len]。
        True 表示可以看，False 表示是 <pad>，不能看。
        """
        return (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """构造目标端 mask，包含 padding mask 和 causal mask。

        padding mask 防止看 <pad>。
        causal mask 防止位置 i 看见 i 之后的 token。

        输入 tgt: [B, tgt_len]
        输出 mask: [B, 1, tgt_len, tgt_len]
        """
        pad_mask = (tgt != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        # 下三角矩阵：第 i 行只能看到 0..i 的列。
        causal = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device, dtype=torch.bool))
        return pad_mask & causal.unsqueeze(0).unsqueeze(1)

    def _reset_parameters(self) -> None:
        """初始化权重。

        Xavier 初始化适合线性层这类矩阵参数，比默认随机初始化更稳定。
        bias 和 LayerNorm 这类一维参数保持 PyTorch 默认初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
