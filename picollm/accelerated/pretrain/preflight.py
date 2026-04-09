import argparse
import json
import os
import time
from dataclasses import asdict

import torch

from picollm.accelerated.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    compute_cleanup,
    compute_init,
    get_peak_flops,
    print0,
)
from picollm.accelerated.flash_attention import USE_FA3
from picollm.accelerated.gpt import GPT, GPTConfig


def build_model_meta(depth: int, aspect_ratio: int, head_dim: int, max_seq_len: int, vocab_size: int, window_pattern: str):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        return GPT(config)


def main():
    parser = argparse.ArgumentParser(description="Fast preflight for the accelerated picoLLM training stack")
    parser.add_argument("--device-type", type=str, default="cuda", help="cuda|cpu|mps")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(args.device_type)
    master_process = ddp_rank == 0
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_name)
        print0(f"Preflight GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    print0(f"Preflight compute dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print0(f"Preflight Flash Attention 3 enabled: {USE_FA3}")

    model = build_model_meta(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.vocab_size,
        args.window_pattern,
    )
    print0(f"Preflight model config:\n{json.dumps(asdict(model.config), indent=2)}")
    model.to_empty(device=device)
    model.init_weights()

    if args.fp8:
        if device.type != "cuda":
            raise SystemExit("FP8 preflight requires CUDA")
        from picollm.accelerated.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)

    compile_disabled = os.environ.get("TORCH_COMPILE_DISABLE") == "1"
    if compile_disabled:
        print0("Preflight torch.compile disabled")
    else:
        model = torch.compile(model, dynamic=False)
        print0("Preflight torch.compile enabled")

    optimizer = model.setup_optimizer()
    model.train()

    x = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, args.vocab_size, (args.device_batch_size, args.max_seq_len), device=device, dtype=torch.long)

    synchronize()
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        synchronize()
        print0(f"Preflight step {step + 1}/{args.steps} | loss: {loss.item():.6f}")
    elapsed = time.time() - t0

    if master_process and device.type == "cuda":
        peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print0(f"Preflight peak memory: {peak_mem_gib:.2f} GiB")
    print0(f"Preflight passed in {elapsed:.2f}s")

    compute_cleanup()


if __name__ == "__main__":
    main()
