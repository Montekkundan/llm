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
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if quant_config is None:
        model = model.to(resolved_device)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        if quant_config is None:
            model = model.to(resolved_device)

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
