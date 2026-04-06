from __future__ import annotations

import torch
import torch.nn as nn


def manual_layer_norm(x: torch.Tensor, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def rms_norm(x: torch.Tensor, eps: float = 1e-5):
    rms = torch.sqrt((x.pow(2).mean(dim=-1, keepdim=True)) + eps)
    return x / rms


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    ln = nn.LayerNorm(4, elementwise_affine=False)
    print('manual == torch LayerNorm:', torch.allclose(manual_layer_norm(x), ln(x), atol=1e-6))
    print('mean after LN:', manual_layer_norm(x).mean(dim=-1))
    print('std after LN :', manual_layer_norm(x).std(dim=-1, unbiased=False))

    near_constant = torch.tensor([[10.0, 10.0, 10.0, 10.0001]])
    print('\nsmall variance input')
    print('eps=1e-5:', manual_layer_norm(near_constant, eps=1e-5))
    print('eps=1e-1:', manual_layer_norm(near_constant, eps=1e-1))

    print('\nRMSNorm keeps mean information:')
    print('LayerNorm:', manual_layer_norm(x)[0])
    print('RMSNorm  :', rms_norm(x)[0])
