from __future__ import annotations

import platform

import torch


def resolve_device(preferred: str = "auto") -> str:
    preferred = preferred.lower()
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype_for_device(device: str) -> torch.dtype:
    device = device.lower()
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def summarize_device(preferred: str = "auto") -> dict[str, str]:
    device = resolve_device(preferred)
    dtype = str(default_dtype_for_device(device)).replace("torch.", "")
    return {
        "requested": preferred,
        "resolved": device,
        "dtype": dtype,
        "platform": platform.platform(),
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "mps_available": str(torch.backends.mps.is_available()).lower(),
    }
