import json
import os
import tempfile
import unittest
from importlib import util as importlib_util
from pathlib import Path
from unittest import mock

import torch

from picollm.accelerated import checkpoint_manager
from picollm.accelerated.gpt import GPT, GPTConfig


class _FakeTokenizer:
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab_size


def _load_restore_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "restore_picollm_from_hf.py"
    spec = importlib_util.spec_from_file_location("restore_picollm_from_hf", module_path)
    module = importlib_util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_checkpoint_layout(
    base_dir: Path,
    source: str,
    *,
    step: int,
    config: GPTConfig,
    model_tag: str = "d2",
    include_optimizer: bool = False,
) -> Path:
    root_name = "base_checkpoints" if source == "base" else "chatsft_checkpoints"
    checkpoint_dir = base_dir / root_name / model_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(config)
    model.init_weights()
    torch.save(model.state_dict(), checkpoint_dir / f"model_{step:06d}.pt")
    with open(checkpoint_dir / f"meta_{step:06d}.json", "w", encoding="utf-8") as handle:
        json.dump({"model_config": config.__dict__}, handle)
    if include_optimizer:
        torch.save({"state": "optim"}, checkpoint_dir / f"optim_{step:06d}_rank0.pt")
    return checkpoint_dir


def _write_tokenizer_layout(base_dir: Path) -> None:
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.pkl").write_text("stub", encoding="utf-8")
    (tokenizer_dir / "token_bytes.pt").write_text("stub", encoding="utf-8")


class CheckpointManagerTests(unittest.TestCase):
    def setUp(self):
        self._old_base_dir = os.environ.get("PICOLLM_BASE_DIR")

    def tearDown(self):
        if self._old_base_dir is None:
            os.environ.pop("PICOLLM_BASE_DIR", None)
        else:
            os.environ["PICOLLM_BASE_DIR"] = self._old_base_dir

    def test_build_model_does_not_call_init_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints" / "d2"
            checkpoint_dir.mkdir(parents=True)
            step = 7

            config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=8)
            model = GPT(config)
            model.init_weights()

            torch.save(model.state_dict(), checkpoint_dir / f"model_{step:06d}.pt")
            with open(checkpoint_dir / f"meta_{step:06d}.json", "w", encoding="utf-8") as handle:
                json.dump({"model_config": config.__dict__}, handle)

            with mock.patch.object(checkpoint_manager, "get_tokenizer", return_value=_FakeTokenizer(config.vocab_size)):
                with mock.patch.object(
                    checkpoint_manager.GPT,
                    "init_weights",
                    side_effect=AssertionError("build_model should not call init_weights() during checkpoint restore"),
                ):
                    restored_model, tokenizer, meta = checkpoint_manager.build_model(
                        checkpoint_dir,
                        step,
                        torch.device("cpu"),
                        phase="eval",
                    )

            self.assertEqual(tokenizer.get_vocab_size(), config.vocab_size)
            self.assertEqual(restored_model.cos.device.type, "cpu")
            self.assertEqual(restored_model.sin.device.type, "cpu")
            self.assertGreaterEqual(restored_model.cos.shape[1], config.sequence_len)
            self.assertEqual(meta["model_config"]["sequence_len"], config.sequence_len)

    def test_find_largest_model_prefers_highest_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "d4").mkdir()
            (root / "d12").mkdir()
            (root / "custom-tag").mkdir()

            self.assertEqual(checkpoint_manager.find_largest_model(str(root)), "d12")

    def test_find_last_step_returns_latest_model_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root.mkdir(parents=True, exist_ok=True)
            for step in (3, 11, 7):
                (root / f"model_{step:06d}.pt").write_text("stub", encoding="utf-8")

            self.assertEqual(checkpoint_manager.find_last_step(str(root)), 11)

    def test_load_optimizer_state_returns_none_when_inference_bundle_has_no_optimizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            os.environ["PICOLLM_BASE_DIR"] = str(base_dir)
            config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=8)
            _write_checkpoint_layout(base_dir, "base", step=5, config=config, include_optimizer=False)

            optimizer_state = checkpoint_manager.load_optimizer_state("base", torch.device("cpu"), rank=0)

            self.assertIsNone(optimizer_state)

    def test_load_model_uses_base_and_sft_layouts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            os.environ["PICOLLM_BASE_DIR"] = str(base_dir)
            config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=8)
            _write_checkpoint_layout(base_dir, "base", step=4, config=config, include_optimizer=True)
            _write_checkpoint_layout(base_dir, "sft", step=9, config=config, include_optimizer=True)

            with mock.patch.object(checkpoint_manager, "get_tokenizer", return_value=_FakeTokenizer(config.vocab_size)):
                base_model, _, base_meta = checkpoint_manager.load_model("base", torch.device("cpu"), phase="eval")
                sft_model, _, sft_meta = checkpoint_manager.load_model("sft", torch.device("cpu"), phase="eval")

            self.assertEqual(base_meta["_checkpoint"]["step"], 4)
            self.assertEqual(sft_meta["_checkpoint"]["step"], 9)
            self.assertEqual(base_meta["_checkpoint"]["model_tag"], "d2")
            self.assertEqual(sft_meta["_checkpoint"]["model_tag"], "d2")
            self.assertEqual(base_model.config.sequence_len, config.sequence_len)
            self.assertEqual(sft_model.config.sequence_len, config.sequence_len)

    def test_restore_layout_loads_via_checkpoint_manager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            os.environ["PICOLLM_BASE_DIR"] = str(base_dir)
            config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=8)
            _write_tokenizer_layout(base_dir)
            _write_checkpoint_layout(base_dir, "base", step=4, config=config, include_optimizer=False)
            _write_checkpoint_layout(base_dir, "sft", step=9, config=config, include_optimizer=False)

            restore_module = _load_restore_module()
            restore_module.verify_layout(base_dir)
            with mock.patch.object(checkpoint_manager, "get_tokenizer", return_value=_FakeTokenizer(config.vocab_size)):
                model, tokenizer, meta = checkpoint_manager.load_model("sft", torch.device("cpu"), phase="eval")

            self.assertEqual(tokenizer.get_vocab_size(), config.vocab_size)
            self.assertEqual(meta["_checkpoint"]["step"], 9)
            self.assertEqual(model.config.n_layer, config.n_layer)


if __name__ == "__main__":
    unittest.main()
