from __future__ import annotations

import json
import shutil
import base64
import importlib
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import __version__ as transformers_version
from transformers.convert_slow_tokenizer import TikTokenConverter

from picollm.accelerated import checkpoint_manager
from picollm.accelerated.common import get_base_dir
from picollm.accelerated.hf_export import PicoLlmConfig
from picollm.accelerated.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


HF_EXPORT_TEMPLATE_DIR = Path(__file__).resolve().parent / "hf_export"


def resolve_export_checkpoint(
    base_dir: Path,
    source: str,
    model_tag: str | None = None,
    step: int | None = None,
) -> tuple[Path, str, int]:
    checkpoint_root = base_dir / {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
    }[source]
    if not checkpoint_root.exists():
        raise SystemExit(f"Missing checkpoint root for {source}: {checkpoint_root}")
    resolved_model_tag = model_tag or checkpoint_manager.find_largest_model(str(checkpoint_root))
    checkpoint_dir = checkpoint_root / resolved_model_tag
    if not checkpoint_dir.exists():
        raise SystemExit(f"Missing checkpoint directory: {checkpoint_dir}")
    resolved_step = step if step is not None else checkpoint_manager.find_last_step(str(checkpoint_dir))
    return checkpoint_dir, resolved_model_tag, resolved_step


def load_export_state(
    checkpoint_dir: Path,
    step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object], PicoLlmConfig]:
    model_state, _, meta = checkpoint_manager.load_checkpoint(
        str(checkpoint_dir),
        step,
        torch.device("cpu"),
        load_optimizer=False,
    )
    model_state = {key.removeprefix("_orig_mod."): value for key, value in model_state.items()}
    model_config_kwargs = dict(meta["model_config"])
    checkpoint_manager._patch_missing_config_keys(model_config_kwargs)
    config = PicoLlmConfig(**model_config_kwargs)
    checkpoint_manager._patch_missing_keys(model_state, config)
    return model_state, meta, config


def normalize_state_dict_for_export(
    model_state: dict[str, torch.Tensor],
    *,
    float_dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        tensor = value.detach().cpu().contiguous()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(float_dtype)
        normalized[key] = tensor
    return normalized


def _special_tokens_map(tokenizer: RustBPETokenizer) -> dict[str, object]:
    bos_token = "<|bos|>"
    eos_token = "<|assistant_end|>"
    pad_token = "<|assistant_end|>"
    extras = [token for token in SPECIAL_TOKENS if token not in {bos_token, eos_token, pad_token}]
    return {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "additional_special_tokens": extras,
        "bos_token_id": tokenizer.encode_special(bos_token),
        "eos_token_id": tokenizer.encode_special(eos_token),
        "pad_token_id": tokenizer.encode_special(pad_token),
    }


def _write_tiktoken_bpe_file(encoding, output_dir: Path) -> Path:
    tiktoken_dir = output_dir / "tiktoken"
    tiktoken_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = tiktoken_dir / "vocab.tiktoken"
    with open(vocab_path, "wb") as handle:
        for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1]):
            handle.write(base64.b64encode(token) + b" " + str(rank).encode("utf-8") + b"\n")
    return vocab_path


def export_tokenizer_to_transformers(
    base_dir: Path,
    output_dir: Path,
    *,
    sequence_len: int,
) -> dict[str, object]:
    tokenizer_dir = base_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise SystemExit(f"Missing tokenizer directory: {tokenizer_dir}")
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))
    vocab_path = _write_tiktoken_bpe_file(tokenizer.enc, output_dir)
    fast_tokenizer = TikTokenConverter(
        vocab_file=str(vocab_path),
        pattern=tokenizer.enc._pat_str,
        extra_special_tokens=tokenizer.enc._special_tokens,
    ).converted()
    fast_tokenizer.save(str(output_dir / "tokenizer.json"))
    special_map = _special_tokens_map(tokenizer)
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": sequence_len,
        "clean_up_tokenization_spaces": False,
        "bos_token": special_map["bos_token"],
        "eos_token": special_map["eos_token"],
        "pad_token": special_map["pad_token"],
        "additional_special_tokens": special_map["additional_special_tokens"],
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": special_map["bos_token"],
                "eos_token": special_map["eos_token"],
                "pad_token": special_map["pad_token"],
                "additional_special_tokens": special_map["additional_special_tokens"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n",
        encoding="utf-8",
    )
    return special_map


def copy_transformers_runtime_files(output_dir: Path) -> None:
    for filename in ("__init__.py", "configuration_picollm.py", "modeling_picollm.py"):
        shutil.copy2(HF_EXPORT_TEMPLATE_DIR / filename, output_dir / filename)


def render_transformers_export_readme(metadata: dict[str, object]) -> str:
    return f"""# picoLLM Transformers Export

This directory is a Transformers-compatible export for the picoLLM checkpoint below.

- Source: `{metadata['source']}`
- Model tag: `{metadata['model_tag']}`
- Step: `{metadata['step']}`
- Export dtype: `{metadata['export_dtype']}`
- Native checkpoint dir: `{metadata['checkpoint_dir']}`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(".", trust_remote_code=True)
```

    This export is an ecosystem bridge. The canonical runtime and training path remains the
    native picoLLM checkpoint format under `PICOLLM_BASE_DIR`.
"""


def export_picollm_to_transformers(
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path,
    source: str = "sft",
    model_tag: str | None = None,
    step: int | None = None,
    export_dtype: str = "float32",
) -> dict[str, object]:
    resolved_base_dir = Path(base_dir or get_base_dir()).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir, resolved_model_tag, resolved_step = resolve_export_checkpoint(
        resolved_base_dir,
        source,
        model_tag=model_tag,
        step=step,
    )
    model_state, meta, config = load_export_state(checkpoint_dir, resolved_step)

    float_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "preserve": None,
    }[export_dtype]
    if float_dtype is None:
        normalized_state = {key: value.detach().cpu().contiguous() for key, value in model_state.items()}
    else:
        normalized_state = normalize_state_dict_for_export(model_state, float_dtype=float_dtype)

    special_map = export_tokenizer_to_transformers(
        resolved_base_dir,
        resolved_output_dir,
        sequence_len=config.sequence_len,
    )
    copy_transformers_runtime_files(resolved_output_dir)

    config_dict = config.to_dict()
    config_dict.update(
        {
            "architectures": ["PicoLlmForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_picollm.PicoLlmConfig",
                "AutoModelForCausalLM": "modeling_picollm.PicoLlmForCausalLM",
            },
            "bos_token_id": special_map["bos_token_id"],
            "eos_token_id": special_map["eos_token_id"],
            "pad_token_id": special_map["pad_token_id"],
            "use_cache": False,
            "torch_dtype": export_dtype if export_dtype != "preserve" else None,
            "transformers_version": transformers_version,
        }
    )
    (resolved_output_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2) + "\n",
        encoding="utf-8",
    )
    (resolved_output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": special_map["bos_token_id"],
                "eos_token_id": special_map["eos_token_id"],
                "pad_token_id": special_map["pad_token_id"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_file(normalized_state, str(resolved_output_dir / "model.safetensors"))

    metadata = {
        "format": "transformers-trust-remote-code",
        "source": source,
        "model_tag": resolved_model_tag,
        "step": resolved_step,
        "export_dtype": export_dtype,
        "checkpoint_dir": str(checkpoint_dir),
        "base_dir": str(resolved_base_dir),
        "model_config": dict(meta["model_config"]),
    }
    (resolved_output_dir / "picollm_transformers_export.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (resolved_output_dir / "README.md").write_text(
        render_transformers_export_readme(metadata),
        encoding="utf-8",
    )
    return metadata


def _resolve_gguf_module(gguf_module=None):
    if gguf_module is not None:
        return gguf_module
    try:
        return importlib.import_module("gguf")
    except ImportError as exc:
        raise SystemExit(
            "GGUF export requires the `gguf` package. Install project dependencies again "
            "after syncing the updated pyproject, or install `gguf>=0.18.0` into the active environment."
        ) from exc


def _gguf_token_payload(tokenizer: RustBPETokenizer, gguf_module) -> tuple[list[bytes], list[object]]:
    normal_type = getattr(getattr(gguf_module, "TokenType", object), "NORMAL", 1)
    control_type = getattr(getattr(gguf_module, "TokenType", object), "CONTROL", 3)
    n_vocab = tokenizer.get_vocab_size()
    tokens: list[bytes] = [b""] * n_vocab
    token_types: list[object] = [normal_type] * n_vocab
    for token_bytes, rank in tokenizer.enc._mergeable_ranks.items():
        tokens[rank] = token_bytes
    for token_name, token_id in tokenizer.enc._special_tokens.items():
        tokens[token_id] = token_name.encode("utf-8")
        token_types[token_id] = control_type
    for token_id, token_bytes in enumerate(tokens):
        if token_bytes:
            continue
        tokens[token_id] = tokenizer.id_to_token(token_id).encode("utf-8")
    return tokens, token_types


def render_gguf_export_readme(metadata: dict[str, object]) -> str:
    return f"""# picoLLM GGUF Export

This directory contains a GGUF export of a native picoLLM checkpoint.

- Source: `{metadata['source']}`
- Model tag: `{metadata['model_tag']}`
- Step: `{metadata['step']}`
- GGUF path: `{metadata['gguf_path']}`

## Important Note

This file is written in GGUF format, but picoLLM is a custom architecture.
Stock llama.cpp does not currently ship a picoLLM runtime implementation, so this export
should be treated as a bridge artifact for future local-runtime work rather than a drop-in
replacement for the native picoLLM checkpoint format today.
"""


def export_picollm_to_gguf(
    *,
    base_dir: str | Path | None = None,
    output_path: str | Path,
    source: str = "sft",
    model_tag: str | None = None,
    step: int | None = None,
    export_dtype: str = "float32",
    gguf_module=None,
) -> dict[str, object]:
    gguf = _resolve_gguf_module(gguf_module=gguf_module)
    resolved_base_dir = Path(base_dir or get_base_dir()).resolve()
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir, resolved_model_tag, resolved_step = resolve_export_checkpoint(
        resolved_base_dir,
        source,
        model_tag=model_tag,
        step=step,
    )
    model_state, _, config = load_export_state(checkpoint_dir, resolved_step)
    float_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "preserve": None,
    }[export_dtype]
    if float_dtype is None:
        normalized_state = {key: value.detach().cpu().contiguous() for key, value in model_state.items()}
    else:
        normalized_state = normalize_state_dict_for_export(model_state, float_dtype=float_dtype)

    tokenizer = RustBPETokenizer.from_directory(str(resolved_base_dir / "tokenizer"))
    special_map = _special_tokens_map(tokenizer)
    tokens, token_types = _gguf_token_payload(tokenizer, gguf)

    writer = gguf.GGUFWriter(str(resolved_output_path), "picollm")
    writer.add_name("picoLLM")
    writer.add_author("picoLLM")
    writer.add_description(
        "picoLLM native GGUF export. Stock llama.cpp does not yet include a picoLLM runtime."
    )
    writer.add_context_length(config.sequence_len)
    writer.add_embedding_length(config.n_embd)
    writer.add_feed_forward_length(4 * config.n_embd)
    writer.add_block_count(config.n_layer)
    writer.add_head_count(config.n_head)
    writer.add_head_count_kv(config.n_kv_head)
    writer.add_vocab_size(config.vocab_size)
    writer.add_causal_attention(True)
    writer.add_tokenizer_model("gpt2-bpe")
    writer.add_tokenizer_pre("picollm-rustbpe")
    writer.add_token_list(tokens)
    if hasattr(writer, "add_token_types"):
        writer.add_token_types(token_types)
    writer.add_bos_token_id(special_map["bos_token_id"])
    writer.add_eos_token_id(special_map["eos_token_id"])
    writer.add_pad_token_id(special_map["pad_token_id"])
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_string("picollm.source", source)
    writer.add_string("picollm.model_tag", resolved_model_tag)
    writer.add_uint32("picollm.step", resolved_step)
    writer.add_string("picollm.window_pattern", config.window_pattern)
    writer.add_string(
        "picollm.runtime_note",
        "Stock llama.cpp does not yet ship picoLLM architecture support.",
    )

    for key, tensor in normalized_state.items():
        writer.add_tensor(key, tensor.numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    metadata = {
        "format": "gguf",
        "architecture": "picollm",
        "source": source,
        "model_tag": resolved_model_tag,
        "step": resolved_step,
        "export_dtype": export_dtype,
        "checkpoint_dir": str(checkpoint_dir),
        "gguf_path": str(resolved_output_path),
    }
    metadata_path = resolved_output_path.with_suffix(resolved_output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme_path = resolved_output_path.with_name(f"{resolved_output_path.stem}.README.md")
    readme_path.write_text(render_gguf_export_readme(metadata), encoding="utf-8")
    return metadata
