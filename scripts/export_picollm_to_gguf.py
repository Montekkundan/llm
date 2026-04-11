#!/usr/bin/env python3
"""Export a picoLLM-native checkpoint into a GGUF file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from picollm.accelerated.common import get_base_dir
from picollm.accelerated.exporters import export_picollm_to_gguf


def default_output_path(base_dir: Path, source: str, model_tag: str | None, step: int | None) -> Path:
    parts = [source]
    if model_tag:
        parts.append(model_tag)
    if step is not None:
        parts.append(f"step{step}")
    suffix = "-".join(parts)
    return base_dir / "exports" / "gguf" / f"{suffix}.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a picoLLM checkpoint to GGUF format.")
    parser.add_argument("--base-dir", type=Path, default=Path(get_base_dir()), help="PICOLLM_BASE_DIR root")
    parser.add_argument("--source", choices=["base", "sft"], default="sft", help="Checkpoint source to export")
    parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint model tag (defaults to latest/largest)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (defaults to latest)")
    parser.add_argument("--output-path", type=Path, default=None, help="GGUF output path")
    parser.add_argument(
        "--export-dtype",
        choices=["float32", "float16", "bfloat16", "preserve"],
        default="float32",
        help="Floating-point dtype used inside the GGUF tensor payload",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else default_output_path(base_dir, args.source, args.model_tag, args.step)
    )
    metadata = export_picollm_to_gguf(
        base_dir=base_dir,
        output_path=output_path,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        export_dtype=args.export_dtype,
    )
    print(f"GGUF export written to: {output_path}")
    print(
        f"Resolved checkpoint: source={metadata['source']} model_tag={metadata['model_tag']} "
        f"step={metadata['step']} dtype={metadata['export_dtype']}"
    )
    print("Note: stock llama.cpp does not yet include picoLLM architecture support.")


if __name__ == "__main__":
    main()
