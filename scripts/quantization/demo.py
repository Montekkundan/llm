from __future__ import annotations

import torch


def symmetric_int8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(x.abs().max() / 127.0) if float(x.abs().max()) > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.float() * scale


if __name__ == "__main__":
    x = torch.tensor([[-1.25, -0.25, 0.0, 0.8], [1.5, 0.3, -0.7, 0.12]], dtype=torch.float32)
    q, scale = symmetric_int8_quantize(x)
    restored = dequantize_int8(q, scale)

    print("float32 bytes:", x.numel() * x.element_size())
    print("int8 bytes   :", q.numel() * q.element_size())
    print("scale        :", round(scale, 6))
    print("original:\n", x)
    print("quantized:\n", q)
    print("restored:\n", restored)
    print("mean abs err :", round(float((x - restored).abs().mean()), 6))
