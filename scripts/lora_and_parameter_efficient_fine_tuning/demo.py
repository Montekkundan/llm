from __future__ import annotations

import math

import torch


def lora_parameter_count(in_features: int, out_features: int, rank: int) -> tuple[int, int]:
    dense = in_features * out_features
    lora = rank * in_features + out_features * rank
    return dense, lora


def low_rank_update_demo(rank: int = 2) -> dict[str, torch.Tensor]:
    base = torch.tensor(
        [
            [0.5, -0.2, 0.1],
            [0.0, 0.3, -0.4],
            [0.7, 0.1, 0.2],
        ],
        dtype=torch.float32,
    )
    target_delta = torch.tensor(
        [
            [0.10, -0.06, 0.00],
            [0.04, 0.02, -0.03],
            [0.08, -0.02, 0.01],
        ],
        dtype=torch.float32,
    )
    u, s, vh = torch.linalg.svd(target_delta)
    u_r = u[:, :rank]
    s_r = torch.diag(torch.sqrt(s[:rank]))
    vh_r = vh[:rank, :]
    a = u_r @ s_r
    b = s_r @ vh_r
    approx_delta = a @ b
    return {"base": base, "target_delta": target_delta, "approx_delta": approx_delta, "adapted": base + approx_delta}


if __name__ == "__main__":
    dense_params, lora_params = lora_parameter_count(in_features=4096, out_features=4096, rank=16)
    print("dense params :", dense_params)
    print("lora params  :", lora_params)
    print("reduction x  :", round(dense_params / lora_params, 2))

    demo = low_rank_update_demo(rank=2)
    print("\nbase weight:\n", demo["base"])
    print("\ntarget delta:\n", demo["target_delta"])
    print("\napprox delta:\n", demo["approx_delta"])
    print("\nrelative error:", round(float(torch.norm(demo["target_delta"] - demo["approx_delta"]) / torch.norm(demo["target_delta"])), 4))
