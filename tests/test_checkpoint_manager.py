import json
import tempfile
import unittest
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


class CheckpointManagerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
