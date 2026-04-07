from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

from .device import default_dtype_for_device, resolve_device, summarize_device


@dataclass
class GenerationBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    model_name_or_path: str
    adapter_path: str | None = None
    quantization: str = "none"


def _maybe_build_quant_config(quantization: str, device: str) -> BitsAndBytesConfig | None:
    quantization = quantization.lower()
    if quantization == "none":
        return None
    if device != "cuda":
        return None
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")


def load_generation_bundle(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    device: str = "auto",
    quantization: str = "none",
    trust_remote_code: bool = False,
) -> GenerationBundle:
    resolved_device = resolve_device(device)
    torch_dtype = default_dtype_for_device(resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = _maybe_build_quant_config(quantization, resolved_device)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch_dtype
        # Keep weights materialized on CPU before moving them to the target device.
        # Some newer Transformers loading paths can leave missing/tied weights on
        # meta tensors under low-memory loading, which then breaks `.to(...)`.
        model_kwargs["low_cpu_mem_usage"] = False
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if quant_config is None:
        model = model.to(resolved_device)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        if quant_config is None:
            model = model.to(resolved_device)

    if getattr(model, "generation_config", None) is not None:
        # Some instruct checkpoints ship sampling defaults that produce noisy warnings
        # during deterministic evaluation. Keep the base config neutral and let
        # generate_reply/stream_reply pass explicit sampling kwargs when needed.
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, field):
                setattr(model.generation_config, field, None)

    model.eval()
    return GenerationBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        model_name_or_path=model_name_or_path,
        adapter_path=str(adapter_path) if adapter_path else None,
        quantization=quantization if _maybe_build_quant_config(quantization, resolved_device) else "none",
    )


def metadata_for_bundle(bundle: GenerationBundle) -> dict[str, object]:
    return {
        "device": summarize_device(bundle.device),
        "model_name_or_path": bundle.model_name_or_path,
        "adapter_path": bundle.adapter_path,
        "quantization": bundle.quantization,
        "parameter_count": sum(parameter.numel() for parameter in bundle.model.parameters()),
    }
