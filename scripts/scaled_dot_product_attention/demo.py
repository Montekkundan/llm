from __future__ import annotations

import math

import torch


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = q @ k.T / scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    return scores, probs, out


if __name__ == "__main__":
    torch.manual_seed(0)

    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    v = torch.tensor([[10.0, 0.0], [7.0, 1.0], [0.0, 10.0]])
    scores, probs, out = manual_attention(q, k, v)
    print("scores:")
    print(scores)
    print("attention probabilities:")
    print(probs)
    print("output:")
    print(out)

    x = torch.randn(4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _, probs, _ = manual_attention(x, x, x, mask=mask)
    print("\ncausal attention probabilities:")
    print(probs)

    for d_k in [4, 16, 64, 256]:
        q = torch.randn(1024, d_k)
        k = torch.randn(1024, d_k)
        raw_logits = (q * k).sum(dim=-1)
        scaled_logits = raw_logits / math.sqrt(d_k)
        print(f"d_k={d_k:>3} | raw std={raw_logits.std():.3f} | scaled std={scaled_logits.std():.3f}")
