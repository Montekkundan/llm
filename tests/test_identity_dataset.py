import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from picollm.accelerated.tasks.customjson import CustomJSON


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "picollm" / "accelerated" / "data" / "identity_conversations.jsonl"
MANIFEST_FILE = REPO_ROOT / "picollm" / "accelerated" / "data" / "identity_conversations.manifest.json"
SPEEDRUN_FILE = REPO_ROOT / "picollm" / "accelerated" / "speedrun.sh"


def load_verify_identity_asset_module():
    module_path = REPO_ROOT / "scripts" / "verify_identity_asset.py"
    spec = importlib.util.spec_from_file_location("verify_identity_asset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_identity_smoke_module():
    module_path = REPO_ROOT / "picollm" / "accelerated" / "chat" / "identity_smoke.py"
    spec = importlib.util.spec_from_file_location("identity_smoke", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IdentityDatasetTests(unittest.TestCase):
    def test_customjson_loads_expected_number_of_rows(self):
        dataset = CustomJSON(filepath=str(DATA_FILE))

        self.assertEqual(dataset.num_examples(), 1000)
        first_example = dataset.get_example(0)
        self.assertIn("messages", first_example)
        self.assertGreaterEqual(len(first_example["messages"]), 2)

    def test_identity_smoke_dataset_only_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "picollm.accelerated.chat.identity_smoke", "--dataset-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("dataset: ok (1000 rows", result.stdout)

    def test_identity_smoke_requires_positive_identity_terms(self):
        identity_smoke = load_identity_smoke_module()

        self.assertEqual(
            identity_smoke.find_missing_expectations(
                "Who created you?",
                "Montek Kundan created picoLLM.",
            ),
            [],
        )
        self.assertNotEqual(
            identity_smoke.find_missing_expectations(
                "Who created you?",
                "A team of researchers created me.",
            ),
            [],
        )
        self.assertEqual(
            identity_smoke.find_missing_expectations(
                "What project are you part of?",
                "I am part of picoLLM in the LLM From Scratch and Deploy repo.",
            ),
            [],
        )

    def test_manifest_matches_canonical_dataset(self):
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        data = DATA_FILE.read_bytes()
        row_count = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())

        self.assertEqual(manifest["row_count"], 1000)
        self.assertEqual(row_count, manifest["row_count"])
        self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["sha256"])

    def test_speedrun_uses_repo_local_identity_file_by_default(self):
        speedrun_text = SPEEDRUN_FILE.read_text(encoding="utf-8")
        self.assertIn(
            'IDENTITY_SOURCE="${PICOLLM_IDENTITY_CONVERSATIONS_FILE:-$REPO_ROOT/picollm/accelerated/data/identity_conversations.jsonl}"',
            speedrun_text,
        )

    def test_hosted_and_local_identity_bytes_must_match(self):
        verify_identity_asset = load_verify_identity_asset_module()
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        local_data = DATA_FILE.read_bytes()

        with mock.patch.object(
            verify_identity_asset,
            "fetch_hosted_asset",
            return_value=local_data,
        ):
            verify_identity_asset.verify_hosted_asset(
                "https://assets.montek.dev/identity_conversations.jsonl",
                local_data,
                manifest["sha256"],
                manifest["row_count"],
            )


if __name__ == "__main__":
    unittest.main()
