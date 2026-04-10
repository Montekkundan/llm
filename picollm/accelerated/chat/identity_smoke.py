import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from picollm.accelerated.engine import Engine


FORBIDDEN_PATTERNS = (
    re.compile(r"\bnanochat[a-z]*\b", re.IGNORECASE),
    re.compile(r"\bnano\b", re.IGNORECASE),
    re.compile(r"\bandrej\b", re.IGNORECASE),
    re.compile(r"\bkarpathy\b", re.IGNORECASE),
)

DEFAULT_PROMPTS = [
    "Who are you?",
    "Who created you?",
    "What project are you part of?",
]


def find_forbidden_terms(text: str) -> list[str]:
    matches: list[str] = []
    for pattern in FORBIDDEN_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(match.group(0))
    return matches


def dataset_check(data_file: Path) -> None:
    text = data_file.read_text(encoding="utf-8")
    forbidden_terms = find_forbidden_terms(text)
    if forbidden_terms:
        unique_terms = ", ".join(sorted(set(term.lower() for term in forbidden_terms)))
        raise SystemExit(f"Identity dataset contains forbidden branding terms: {unique_terms}")

    rows = [line for line in text.splitlines() if line.strip()]
    for row in rows:
        messages = json.loads(row)
        assert isinstance(messages, list) and messages

    print(f"dataset: ok ({len(rows)} rows, no forbidden branding terms)")


def build_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    tokens = [bos, user_start]
    tokens.extend(tokenizer.encode(prompt))
    tokens.extend([user_end, assistant_start])
    return tokens


def generate_answer(engine: "Engine", tokenizer, prompt: str, max_tokens: int, seed: int) -> str:
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    generated: list[int] = []
    for token_column, _ in engine.generate(
        build_prompt_tokens(tokenizer, prompt),
        num_samples=1,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        seed=seed,
    ):
        token = token_column[0]
        if token in {assistant_end, bos}:
            break
        generated.append(token)
    return tokenizer.decode(generated).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run branding smoke checks against the latest picoLLM checkpoint")
    parser.add_argument("-i", "--source", type=str, default="sft", choices=["base", "sft"], help="Source of the model: base|sft")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type: cuda|cpu|mps. empty => autodetect")
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum number of tokens to generate for each branding prompt")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic smoke-test generation")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "identity_conversations.jsonl",
        help="Identity dataset file to validate before loading the model",
    )
    parser.add_argument("--dataset-only", action="store_true", help="Only validate the identity dataset and skip model loading")
    parser.add_argument("--skip-dataset-check", action="store_true", help="Skip the dataset branding scan")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_dataset_check:
        dataset_check(args.data_file)

    if args.dataset_only:
        return 0

    from picollm.accelerated.checkpoint_manager import load_model
    from picollm.accelerated.common import autodetect_device_type, compute_init
    from picollm.accelerated.engine import Engine

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    failures: list[str] = []
    for prompt in DEFAULT_PROMPTS:
        answer = generate_answer(engine, tokenizer, prompt, max_tokens=args.max_tokens, seed=args.seed)
        forbidden_terms = find_forbidden_terms(answer)
        print(f"Q: {prompt}")
        print(f"A: {answer}\n")
        if forbidden_terms:
            failures.append(f"{prompt} -> {', '.join(sorted(set(term.lower() for term in forbidden_terms)))}")

    if failures:
        raise SystemExit("Branding smoke test failed:\n" + "\n".join(failures))

    print("identity smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
