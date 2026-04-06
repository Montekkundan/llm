from __future__ import annotations

import math

import torch


def self_attention(x: torch.Tensor, bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    scores = x @ x.T / math.sqrt(x.size(-1))
    if bias is not None:
        scores = scores + bias
    probs = torch.softmax(scores, dim=-1)
    return probs @ x, probs


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dim = x.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return x * cos + rotate_half(x) * sin


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    perm = torch.tensor([2, 0, 3, 1])
    out, _ = self_attention(x)
    out_perm, _ = self_attention(x[perm])
    restored = torch.empty_like(out_perm)
    restored[perm] = out_perm
    print("permutation-equivariant without positions:", torch.allclose(out, restored, atol=1e-6))

    pe = sinusoidal_positions(max_len=4, d_model=4)
    out_with_pos, _ = self_attention(x + pe)
    out_perm_with_fixed_pos, _ = self_attention(x[perm] + pe)
    restored = torch.empty_like(out_perm_with_fixed_pos)
    restored[perm] = out_perm_with_fixed_pos
    print("permutation-equivariant with fixed positions:", torch.allclose(out_with_pos, restored, atol=1e-6))

    q = torch.tensor([[1.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.5, 0.5]])
    positions = torch.tensor([0, 3])
    q_rot = apply_rope(q, positions)
    print("RoPE-rotated vectors:")
    print(q_rot)
