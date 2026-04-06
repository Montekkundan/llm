from __future__ import annotations

import math

import torch
import torch.nn as nn


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    probs = torch.softmax(scores, dim=-1)
    return probs @ v, probs


def kv_cache_elements(batch: int, seq_len: int, num_kv_heads: int, head_dim: int, layers: int) -> int:
    return batch * seq_len * num_kv_heads * head_dim * 2 * layers


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(1, 4, 8)
    num_heads = 2
    q_proj = nn.Linear(8, 8, bias=False)
    k_proj = nn.Linear(8, 8, bias=False)
    v_proj = nn.Linear(8, 8, bias=False)

    q = split_heads(q_proj(x), num_heads)
    k = split_heads(k_proj(x), num_heads)
    v = split_heads(v_proj(x), num_heads)
    head_out, head_probs = scaled_dot_product_attention(q, k, v)
    merged = merge_heads(head_out)

    print("per-head attention maps:")
    print(head_probs[0, 0])
    print()
    print(head_probs[0, 1])
    print("\nmerged output shape:", tuple(merged.shape))

    for label, num_kv_heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
        elems = kv_cache_elements(batch=1, seq_len=8192, num_kv_heads=num_kv_heads, head_dim=128, layers=24)
        print(f"{label}: {elems:,} float values")
