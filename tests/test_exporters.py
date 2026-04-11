import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from picollm.accelerated.exporters import (
    export_picollm_to_gguf,
    export_picollm_to_transformers,
)
from picollm.accelerated.gpt import GPT, GPTConfig
from picollm.accelerated.tokenizer import RustBPETokenizer


def _write_checkpoint_layout(
    base_dir: Path,
    source: str,
    *,
    step: int,
    config: GPTConfig,
    model_tag: str = "d2",
) -> Path:
    root_name = "base_checkpoints" if source == "base" else "chatsft_checkpoints"
    checkpoint_dir = base_dir / root_name / model_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = GPT(config)
    model.init_weights()
    torch.save(model.state_dict(), checkpoint_dir / f"model_{step:06d}.pt")
    with open(checkpoint_dir / f"meta_{step:06d}.json", "w", encoding="utf-8") as handle:
        json.dump({"model_config": config.__dict__}, handle)
    return checkpoint_dir


def _write_tokenizer_layout(base_dir: Path) -> None:
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = RustBPETokenizer.train_from_iterator(
        [
            "picoLLM writes its own identity data",
            "this tokenizer export should round-trip through transformers",
            "hello world from a tiny test corpus",
        ],
        vocab_size=320,
    )
    tokenizer.save(str(tokenizer_dir))


class ExportersTests(unittest.TestCase):
    def test_transformers_export_loads_via_auto_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "artifacts"
            output_dir = Path(temp_dir) / "transformers-export"
            config = GPTConfig(
                sequence_len=16,
                vocab_size=320,
                n_layer=2,
                n_head=4,
                n_kv_head=4,
                n_embd=32,
                window_pattern="SL",
            )
            _write_tokenizer_layout(base_dir)
            _write_checkpoint_layout(base_dir, "sft", step=9, config=config)

            metadata = export_picollm_to_transformers(
                base_dir=base_dir,
                output_dir=output_dir,
                source="sft",
            )

            self.assertEqual(metadata["source"], "sft")
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "model.safetensors").exists())
            self.assertTrue((output_dir / "tokenizer.json").exists())
            self.assertTrue((output_dir / "modeling_picollm.py").exists())

            exported_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
            self.assertEqual(exported_config.model_type, "picollm")
            self.assertEqual(exported_config.vocab_size, config.vocab_size)

            tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

            encoded = tokenizer("hello world", return_tensors="pt")
            outputs = model(**encoded)
            self.assertEqual(outputs.logits.shape[0], 1)
            self.assertEqual(outputs.logits.shape[1], encoded["input_ids"].shape[1])
            self.assertEqual(outputs.logits.shape[2], config.vocab_size)

            generated = model.generate(**encoded, max_new_tokens=1)
            self.assertEqual(generated.shape[0], 1)
            self.assertEqual(generated.shape[1], encoded["input_ids"].shape[1] + 1)

    def test_gguf_export_writes_metadata_with_fake_writer(self):
        class _FakeTokenType:
            NORMAL = "normal"
            CONTROL = "control"

        class _FakeGGUFWriter:
            def __init__(self, path, arch):
                self.path = Path(path)
                self.arch = arch
                self.calls = []

            def _record(self, name, *args):
                self.calls.append((name, args))

            def add_name(self, value): self._record("add_name", value)
            def add_author(self, value): self._record("add_author", value)
            def add_description(self, value): self._record("add_description", value)
            def add_context_length(self, value): self._record("add_context_length", value)
            def add_embedding_length(self, value): self._record("add_embedding_length", value)
            def add_feed_forward_length(self, value): self._record("add_feed_forward_length", value)
            def add_block_count(self, value): self._record("add_block_count", value)
            def add_head_count(self, value): self._record("add_head_count", value)
            def add_head_count_kv(self, value): self._record("add_head_count_kv", value)
            def add_vocab_size(self, value): self._record("add_vocab_size", value)
            def add_causal_attention(self, value): self._record("add_causal_attention", value)
            def add_tokenizer_model(self, value): self._record("add_tokenizer_model", value)
            def add_tokenizer_pre(self, value): self._record("add_tokenizer_pre", value)
            def add_token_list(self, value): self._record("add_token_list", len(value))
            def add_token_types(self, value): self._record("add_token_types", len(value))
            def add_bos_token_id(self, value): self._record("add_bos_token_id", value)
            def add_eos_token_id(self, value): self._record("add_eos_token_id", value)
            def add_pad_token_id(self, value): self._record("add_pad_token_id", value)
            def add_add_bos_token(self, value): self._record("add_add_bos_token", value)
            def add_add_eos_token(self, value): self._record("add_add_eos_token", value)
            def add_string(self, key, value): self._record("add_string", key, value)
            def add_uint32(self, key, value): self._record("add_uint32", key, value)
            def add_tensor(self, key, value): self._record("add_tensor", key, tuple(value.shape))
            def write_header_to_file(self): self.path.write_bytes(b"GGUF")
            def write_kv_data_to_file(self): self._record("write_kv_data_to_file")
            def write_ti_data_to_file(self): self._record("write_ti_data_to_file")
            def write_tensors_to_file(self): self._record("write_tensors_to_file")
            def close(self): self._record("close")

        class _FakeGGUFModule:
            TokenType = _FakeTokenType
            GGUFWriter = _FakeGGUFWriter

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "artifacts"
            output_path = Path(temp_dir) / "export.gguf"
            config = GPTConfig(
                sequence_len=16,
                vocab_size=320,
                n_layer=2,
                n_head=4,
                n_kv_head=4,
                n_embd=32,
                window_pattern="SL",
            )
            _write_tokenizer_layout(base_dir)
            _write_checkpoint_layout(base_dir, "sft", step=9, config=config)

            metadata = export_picollm_to_gguf(
                base_dir=base_dir,
                output_path=output_path,
                source="sft",
                gguf_module=_FakeGGUFModule,
            )

            self.assertEqual(metadata["architecture"], "picollm")
            self.assertTrue(output_path.exists())
            self.assertTrue(Path(str(output_path) + ".json").exists())
            self.assertTrue(output_path.with_name(f"{output_path.stem}.README.md").exists())


if __name__ == "__main__":
    unittest.main()
