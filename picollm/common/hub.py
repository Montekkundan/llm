from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from huggingface_hub import HfApi, snapshot_download


def _yaml_list_block(name: str, values: list[str]) -> str:
    if not values:
        return ""
    lines = [f"{name}:"]
    lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def write_model_card(
    folder_path: str | Path,
    repo_id: str,
    *,
    title: str | None = None,
    summary: str | None = None,
    base_model: str | None = None,
    datasets: list[str] | None = None,
    wandb_url: str | None = None,
    license_name: str = "mit",
    pipeline_tag: str = "text-generation",
    library_name: str = "transformers",
    tags: list[str] | None = None,
) -> Path:
    folder = Path(folder_path)
    datasets = datasets or []
    tags = tags or []
    default_tags = ["picollm", "conversational", "transformers", "safetensors"]
    merged_tags: list[str] = []
    for tag in [*default_tags, *tags]:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)

    card_title = title or repo_id.split("/")[-1]
    card_summary = summary or (
        "A conversational GPT-style checkpoint trained in the picoLLM course workflow and "
        "pushed to the Hugging Face Hub for local testing and student reuse."
    )

    yaml_lines = [
        "---",
        "language:",
        "- en",
        f"license: {license_name}",
        f"library_name: {library_name}",
        f"pipeline_tag: {pipeline_tag}",
    ]
    if base_model:
        yaml_lines.append(f"base_model: {base_model}")
    datasets_block = _yaml_list_block("datasets", datasets)
    if datasets_block:
        yaml_lines.append(datasets_block)
    tags_block = _yaml_list_block("tags", merged_tags)
    if tags_block:
        yaml_lines.append(tags_block)
    yaml_lines.append("---")

    sections = [
        "\n".join(yaml_lines),
        f"# {card_title}",
        card_summary,
        "## What this model is",
        dedent(
            f"""\
            This repository contains a GPT-style chat checkpoint produced by the `picollm` training workflow.
            It is intended as a course and experimentation model: small enough to inspect, easy to serve locally,
            and simple to compare against earlier checkpoints or later post-training runs.
            """
        ).strip(),
        "## Training recipe",
        dedent(
            """\
            The model was produced in two stages:

            1. from-scratch decoder-only pretraining on general text
            2. supervised chat fine-tuning on a small conversational dataset

            The practical training code lives in the `picollm/` folder of the companion repository:
            [Montekkundan/llm](https://github.com/Montekkundan/llm/tree/main/picollm)
            """
        ).strip(),
        "## Intended use",
        dedent(
            """\
            Use this checkpoint for:

            - local chat demos
            - model-comparison exercises
            - course walkthroughs on serving, evaluation, and deployment

            Do not treat it as a production safety-reviewed assistant.
            """
        ).strip(),
        "## Quick start",
        dedent(
            f"""\
            ```bash
            uv run python -m picollm.serve.chat_cli \\
              --model {repo_id} \\
              --device auto
            ```

            Or with Transformers:

            ```python
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "{repo_id}"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            ```
            """
        ).strip(),
        "## Notes",
        dedent(
            """\
            This page uses a custom model card so the repository stays readable and course-oriented.
            If you retrain or retune the model, update the summary, datasets, and links before sharing it widely.
            """
        ).strip(),
    ]
    if base_model:
        sections.insert(4, f"Base checkpoint: `{base_model}`")
    if datasets:
        sections.insert(5, "Datasets: " + ", ".join(f"`{name}`" for name in datasets))
    if wandb_url:
        sections.insert(6, f"Training dashboard: [Weights & Biases run]({wandb_url})")

    content = "\n\n".join(section for section in sections if section).strip() + "\n"
    readme_path = folder / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    commit_message: str = "Upload model artifacts",
    private: bool = False,
) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def download_snapshot(repo_id: str, revision: str | None = None, local_dir: str | Path | None = None) -> str:
    return snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(local_dir) if local_dir else None)
