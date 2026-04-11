#!/usr/bin/env python3
"""Export a picoLLM-native checkpoint into a Transformers-compatible package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picollm.accelerated.common import get_base_dir
from picollm.accelerated.exporters import export_picollm_to_transformers


def default_output_dir(base_dir: Path, source: str, model_tag: str | None, step: int | None) -> Path:
    parts = [source]
    if model_tag:
        parts.append(model_tag)
    if step is not None:
        parts.append(f"step{step}")
    suffix = "-".join(parts)
    return base_dir / "exports" / "transformers" / suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a picoLLM checkpoint to a Transformers trust_remote_code package."
    )
    parser.add_argument("--base-dir", type=Path, default=Path(get_base_dir()), help="PICOLLM_BASE_DIR root")
    parser.add_argument("--source", choices=["base", "sft"], default="sft", help="Checkpoint source to export")
    parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint model tag (defaults to latest/largest)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (defaults to latest)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Export directory")
    parser.add_argument(
        "--export-dtype",
        choices=["float32", "float16", "bfloat16", "preserve"],
        default="float32",
        help="Floating-point dtype used in model.safetensors",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(base_dir, args.source, args.model_tag, args.step)
    )
    metadata = export_picollm_to_transformers(
        base_dir=base_dir,
        output_dir=output_dir,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        export_dtype=args.export_dtype,
    )
    print(f"Transformers export written to: {output_dir}")
    print(
        f"Resolved checkpoint: source={metadata['source']} model_tag={metadata['model_tag']} "
        f"step={metadata['step']} dtype={metadata['export_dtype']}"
    )


if __name__ == "__main__":
    main()
