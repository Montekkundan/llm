from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from datasets import Dataset, IterableDataset, load_dataset

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
