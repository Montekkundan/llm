from __future__ import annotations

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = 8, n_heads: int = 2, d_ff: int = 32):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x, attn_weights


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    block = EncoderBlock()
    y, weights = block(x)
    print('shape preserved:', x.shape, '->', y.shape)
    print('attention weights shape:', weights.shape)
    pad_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])
    _, masked = block(x, key_padding_mask=pad_mask)
    print('masked last column in batch 0 head 0:')
    print(masked[0, 0, :, -1])
