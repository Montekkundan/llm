from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .text_format import normalize_text


TextDataset = Dataset | IterableDataset


def _dataset_columns(dataset: TextDataset) -> list[str] | None:
    if getattr(dataset, "column_names", None):
        return list(dataset.column_names)
    if getattr(dataset, "features", None):
        return list(dataset.features.keys())
    return None


def load_text_dataset(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    text_files: list[str],
    alternating_chat_roles: bool,
    streaming: bool = False,
    output_text_column: str = "text",
) -> TextDataset:
    if dataset_name:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
        column_names = _dataset_columns(dataset)

        def _normalize(example: dict[str, object]) -> dict[str, str]:
            return {output_text_column: normalize_text(example[text_column], alternating_chat_roles)}

        if streaming:
            return dataset.map(_normalize, remove_columns=column_names).filter(lambda item: bool(item[output_text_column]))

        return dataset.map(_normalize, remove_columns=column_names).filter(lambda item: bool(item[output_text_column]))

    rows: list[dict[str, str]] = []
    for file_path in text_files:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append({output_text_column: line})
    return Dataset.from_list(rows)


def _message_rows_to_prompt_completion(value: object, eos_token: str | None) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None

    rendered: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        rendered.append((role, content))

    if len(rendered) < 2 or rendered[-1][0] != "assistant":
        return None

    prompt_lines = [f"<|{role}|> {content}" for role, content in rendered[:-1]]
    prompt_lines.append("<|assistant|>")
    completion = f" {rendered[-1][1]}"
    if eos_token:
        completion = f"{completion}{eos_token}"
    return {
        "prompt": "\n".join(prompt_lines),
        "completion": completion,
    }


def load_prompt_completion_dataset(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    messages_column: str,
    eos_token: str | None = None,
    streaming: bool = False,
) -> TextDataset:
    if not dataset_name:
        raise ValueError("A dataset_name is required for prompt-completion loading.")

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    column_names = _dataset_columns(dataset)

    def _normalize(example: dict[str, object]) -> dict[str, str] | None:
        return _message_rows_to_prompt_completion(example[messages_column], eos_token)

    mapped = dataset.map(_normalize, remove_columns=column_names)
    return mapped.filter(lambda item: bool(item["prompt"]) and bool(item["completion"]))


def load_tokenized_chat_dataset(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    messages_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int | None = None,
    streaming: bool = False,
) -> TextDataset:
    if not dataset_name:
        raise ValueError("A dataset_name is required for tokenized chat loading.")

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    column_names = _dataset_columns(dataset)
    eos_token = tokenizer.eos_token or ""

    def _tokenize_example(example: dict[str, object]) -> dict[str, list[int]] | None:
        row = _message_rows_to_prompt_completion(example[messages_column], eos_token)
        if row is None:
            return None

        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        labels = ([-100] * len(prompt_ids)) + completion_ids[:]
        attention_mask = [1] * len(input_ids)

        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
            attention_mask = attention_mask[-max_length:]

        if not any(label != -100 for label in labels):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    mapped = dataset.map(_tokenize_example, remove_columns=column_names)
    return mapped.filter(lambda item: bool(item["input_ids"]) and any(label != -100 for label in item["labels"]))


def iter_texts(
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
    text_files: list[str],
    alternating_chat_roles: bool,
    streaming: bool = False,
    max_texts: int | None = None,
    output_text_column: str = "text",
) -> Iterator[str]:
    dataset = load_text_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        text_column=text_column,
        text_files=text_files,
        alternating_chat_roles=alternating_chat_roles,
        streaming=streaming,
        output_text_column=output_text_column,
    )
    iterator = (row[output_text_column] for row in dataset if row[output_text_column])
    if max_texts is not None:
        iterator = islice(iterator, max_texts)
    return iterator
