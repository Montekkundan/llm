import argparse
import json
import os
import shlex

import torch


SEQ_LEN = 2048
REF_TOTAL_BATCH_SIZE = 1_048_576


def _parse_int_env(name: str):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be > 0, got {parsed}")
    return parsed


def _parse_bool_env(name: str):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return 1
    if value in {"0", "false", "no", "off"}:
        return 0
    raise SystemExit(f"{name} must be a boolean-like value, got {value!r}")


def _parse_choice_env(name: str, allowed: set[str]):
    value = os.environ.get(name, "").strip().upper()
    if not value:
        return None
    if value not in allowed:
        raise SystemExit(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _choose_device_batch_size(min_memory_gib: float, all_hopper: bool) -> int:
    if all_hopper:
        if min_memory_gib >= 70:
            return 16
        if min_memory_gib >= 45:
            return 8
        if min_memory_gib >= 30:
            return 4
        if min_memory_gib >= 22:
            return 2
        return 1

    if min_memory_gib >= 70:
        return 8
    if min_memory_gib >= 45:
        return 4
    if min_memory_gib >= 30:
        return 2
    return 1


def _choose_total_batch_size(nproc_per_node: int, device_batch_size: int, enable_fp8: bool) -> int:
    world_tokens_per_microbatch = nproc_per_node * device_batch_size * SEQ_LEN
    accum_cap = 16 if enable_fp8 else 8
    return max(world_tokens_per_microbatch, min(REF_TOTAL_BATCH_SIZE, world_tokens_per_microbatch * accum_cap))


def _detect_config():
    if not torch.cuda.is_available():
        return {
            "PICOLLM_SUPPORTED": 0,
            "PICOLLM_UNSUPPORTED_REASON": "speedrun.sh currently expects at least one visible CUDA GPU",
            "PICOLLM_GPU_COUNT": 0,
            "PICOLLM_GPU_NAME": "none",
            "PICOLLM_MIN_GPU_MEMORY_GIB": "0.0",
            "PICOLLM_MIN_GPU_CAPABILITY": "none",
            "PICOLLM_HETEROGENEOUS_GPUS": 0,
            "PICOLLM_NPROC_PER_NODE": 1,
            "PICOLLM_DEVICE_BATCH_SIZE": 1,
            "PICOLLM_TOTAL_BATCH_SIZE": 16_384,
            "PICOLLM_ACTIVATION_CHECKPOINTING": 1,
            "PICOLLM_ENABLE_FP8": 0,
            "PICOLLM_WINDOW_PATTERN": "L",
            "PICOLLM_CHAT_EVAL_BATCH_SIZE": 1,
            "PICOLLM_FORCE_FLASH_IMPL": "sdpa",
            "PICOLLM_HARDWARE_SUMMARY": "no visible CUDA GPUs",
        }

    gpu_count = torch.cuda.device_count()
    names = []
    memories_gib = []
    capabilities = []
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        names.append(props.name)
        memories_gib.append(props.total_memory / (1024 ** 3))
        capabilities.append(torch.cuda.get_device_capability(idx))

    min_memory_gib = min(memories_gib)
    all_hopper = all(major >= 9 for major, _ in capabilities)
    heterogeneous = len(set(zip(names, capabilities, [round(v, 1) for v in memories_gib]))) > 1

    auto_nproc_per_node = gpu_count
    auto_enable_fp8 = 1 if all_hopper else 0
    auto_device_batch_size = _choose_device_batch_size(min_memory_gib, all_hopper)
    auto_activation_checkpointing = 0 if all_hopper and min_memory_gib >= 70 else 1
    auto_window_pattern = "SSSL" if all_hopper else "L"

    nproc_per_node = _parse_int_env("PICOLLM_NPROC_PER_NODE") or auto_nproc_per_node
    if nproc_per_node > gpu_count:
        raise SystemExit(
            f"PICOLLM_NPROC_PER_NODE={nproc_per_node} exceeds visible CUDA device count ({gpu_count})"
        )

    enable_fp8 = _parse_bool_env("PICOLLM_ENABLE_FP8")
    if enable_fp8 is None:
        enable_fp8 = auto_enable_fp8
    if enable_fp8 and not all_hopper:
        raise SystemExit(
            "PICOLLM_ENABLE_FP8=1 requires Hopper-or-newer GPUs on every visible rank"
        )

    device_batch_size = _parse_int_env("PICOLLM_DEVICE_BATCH_SIZE") or auto_device_batch_size
    activation_checkpointing = _parse_bool_env("PICOLLM_ACTIVATION_CHECKPOINTING")
    if activation_checkpointing is None:
        activation_checkpointing = auto_activation_checkpointing

    window_pattern = _parse_choice_env("PICOLLM_WINDOW_PATTERN", {"L", "S", "SSSL"})
    if window_pattern is None:
        window_pattern = auto_window_pattern

    total_batch_size = _parse_int_env("PICOLLM_TOTAL_BATCH_SIZE")
    if total_batch_size is None:
        total_batch_size = _choose_total_batch_size(nproc_per_node, device_batch_size, bool(enable_fp8))

    world_tokens_per_microbatch = nproc_per_node * device_batch_size * SEQ_LEN
    if total_batch_size % world_tokens_per_microbatch != 0:
        raise SystemExit(
            "PICOLLM_TOTAL_BATCH_SIZE must be divisible by "
            f"nproc_per_node * device_batch_size * {SEQ_LEN} (= {world_tokens_per_microbatch})"
        )

    chat_eval_batch_size = max(1, min(8, device_batch_size))
    force_flash_impl = "auto" if all_hopper else "sdpa"

    if heterogeneous:
        gpu_summary = "; ".join(
            f"gpu{idx}={name} ({mem:.1f} GiB, sm{major}{minor})"
            for idx, (name, mem, (major, minor)) in enumerate(zip(names, memories_gib, capabilities))
        )
    else:
        major, minor = capabilities[0]
        gpu_summary = f"{gpu_count}x {names[0]} ({min_memory_gib:.1f} GiB each, sm{major}{minor})"

    settings_summary = (
        f"nproc={nproc_per_node}, batch={device_batch_size}, total_batch={total_batch_size}, "
        f"fp8={enable_fp8}, activation_checkpointing={activation_checkpointing}, "
        f"window_pattern={window_pattern}, flash_impl={force_flash_impl}"
    )

    min_major, min_minor = min(capabilities)
    return {
        "PICOLLM_SUPPORTED": 1,
        "PICOLLM_UNSUPPORTED_REASON": "",
        "PICOLLM_GPU_COUNT": gpu_count,
        "PICOLLM_GPU_NAME": names[0] if not heterogeneous else "heterogeneous",
        "PICOLLM_MIN_GPU_MEMORY_GIB": f"{min_memory_gib:.1f}",
        "PICOLLM_MIN_GPU_CAPABILITY": f"{min_major}.{min_minor}",
        "PICOLLM_HETEROGENEOUS_GPUS": 1 if heterogeneous else 0,
        "PICOLLM_NPROC_PER_NODE": nproc_per_node,
        "PICOLLM_DEVICE_BATCH_SIZE": device_batch_size,
        "PICOLLM_TOTAL_BATCH_SIZE": total_batch_size,
        "PICOLLM_ACTIVATION_CHECKPOINTING": activation_checkpointing,
        "PICOLLM_ENABLE_FP8": enable_fp8,
        "PICOLLM_WINDOW_PATTERN": window_pattern,
        "PICOLLM_CHAT_EVAL_BATCH_SIZE": chat_eval_batch_size,
        "PICOLLM_FORCE_FLASH_IMPL": force_flash_impl,
        "PICOLLM_HARDWARE_SUMMARY": gpu_summary,
        "PICOLLM_SETTINGS_SUMMARY": settings_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect hardware-aware defaults for picoLLM speedrun")
    parser.add_argument("--format", choices=["shell", "json"], default="shell")
    args = parser.parse_args()

    config = _detect_config()
    if args.format == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    for key, value in config.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
