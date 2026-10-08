import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None], persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, q, k, start_pos=0):
        """
        q, k : [batch, heads, seq_len, head_dim]
        start_pos : offset for incremental decoding (KV cache)
        """
        end_q = start_pos + q.shape[2]
        end_k = start_pos + k.shape[2]
        needed = max(end_q, end_k)
        if needed > self.max_seq_len_cached:
            self._build_cache(needed * 2)

        cos_q = self.cos_cached[:, :, start_pos:end_q]
        sin_q = self.sin_cached[:, :, start_pos:end_q]
        cos_k = self.cos_cached[:, :, start_pos:end_k]
        sin_k = self.sin_cached[:, :, start_pos:end_k]

        q = q * cos_q + rotate_half(q) * sin_q
        k = k * cos_k + rotate_half(k) * sin_k
        return q, k