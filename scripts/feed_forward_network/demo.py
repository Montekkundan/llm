from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden)
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    ffn = StandardFFN(d_model=8, d_ff=32)
    y = ffn(x)
    print('input shape :', tuple(x.shape))
    print('output shape:', tuple(y.shape))
    print('same FFN row-by-row:', torch.allclose(y[0, 0], ffn(x[0, 0:1])[0]))

    relu = F.relu(x)
    gelu = F.gelu(x)
    silu = F.silu(x)
    print('\nactivation means')
    print('relu:', relu.mean().item())
    print('gelu:', gelu.mean().item())
    print('silu:', silu.mean().item())

    standard = StandardFFN(512, 2048)
    swiglu = SwiGLUFFN(512, int(2048 * 2 / 3))
    print('\nparameter count')
    print('standard:', sum(p.numel() for p in standard.parameters()))
    print('swiglu  :', sum(p.numel() for p in swiglu.parameters()))
