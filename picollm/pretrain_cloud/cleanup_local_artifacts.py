from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def validate_target(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed_root = Path.cwd().resolve() / "artifacts" / "picollm"
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise SystemExit(
            f"Refusing to delete {resolved}. Cleanup targets must stay inside {allowed_root}."
        )
    return resolved


def remove_path(path: Path) -> None:
    if not path.exists():
        print(f"skip: {path} does not exist")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove copied local cloud-training artifacts under artifacts/picollm."
    )
    parser.add_argument("--checkpoint-dir", default="artifacts/picollm/pretrain-run")
    parser.add_argument("--tokenizer-dir", default="artifacts/picollm/tokenizer")
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Also remove the local tokenizer directory.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    targets = [validate_target(Path(args.checkpoint_dir))]
    if args.include_tokenizer:
        targets.append(validate_target(Path(args.tokenizer_dir)))

    if not args.yes:
        print("This will remove the following local paths:")
        for target in targets:
            print(f"- {target}")
        answer = input("Continue? [y/N] ").strip()
        if answer.lower() not in {"y", "yes"}:
            raise SystemExit("Cancelled.")

    for target in targets:
        remove_path(target)


if __name__ == "__main__":
    main()
