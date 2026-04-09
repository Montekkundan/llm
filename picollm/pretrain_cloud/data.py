from __future__ import annotations
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from picollm.common.conversation import encode_messages_for_training

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
    def _tokenize_example(example: dict[str, object]) -> dict[str, list[int]] | None:
        encoded = encode_messages_for_training(
            tokenizer,
            example[messages_column],  # type: ignore[arg-type]
            max_length=max_length,
        )
        return encoded

    mapped = dataset.map(_tokenize_example, remove_columns=column_names)
    return mapped.filter(lambda item: bool(item["input_ids"]) and any(label != -100 for label in item["labels"]))


def _repeat_dataset(dataset: Dataset, repeats: int) -> Dataset:
    repeats = max(1, int(repeats))
    if repeats == 1:
        return dataset
    return concatenate_datasets([dataset] * repeats)


def _tokenize_message_example(
    tokenizer: PreTrainedTokenizerBase,
    messages: object,
    *,
    max_length: int | None = None,
) -> dict[str, list[int]] | None:
    return encode_messages_for_training(
        tokenizer,
        messages,  # type: ignore[arg-type]
        max_length=max_length,
    )


def _tokenize_messages_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int | None = None,
    messages_column: str = "messages",
) -> Dataset:
    def _tokenize_example(example: dict[str, object]) -> dict[str, list[int]] | None:
        return _tokenize_message_example(
            tokenizer,
            example[messages_column],
            max_length=max_length,
        )

    mapped = dataset.map(_tokenize_example, remove_columns=dataset.column_names)
    return mapped.filter(lambda item: bool(item["input_ids"]) and any(label != -100 for label in item["labels"]))


def _choice_letters(count: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [alphabet[index] for index in range(count)]


def _build_identity_conversations() -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {
                    "role": "assistant",
                    "content": (
                        "I am picoLLM, a small teaching-oriented language model built from scratch "
                        "for learning about tokenization, embeddings, attention, fine-tuning, and deployment."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What are you good at?"},
                {
                    "role": "assistant",
                    "content": (
                        "I am best at explaining core machine learning and transformer concepts in a clear, practical way."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What topics can you help me study?"},
                {
                    "role": "assistant",
                    "content": (
                        "I can help with tokenization, embeddings, positional encodings, attention, transformers, "
                        "optimization, fine-tuning, and model deployment."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you be concise?"},
                {
                    "role": "assistant",
                    "content": "Yes. I can answer briefly first and add detail only when it helps.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain things like a teacher, not like marketing copy."},
                {
                    "role": "assistant",
                    "content": (
                        "Understood. I will keep the explanation direct, technical, and focused on what matters."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the goal of picoLLM?"},
                {
                    "role": "assistant",
                    "content": (
                        "The goal is to make language models easier to understand by building and training them in a transparent way."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Should knowledge be free?"},
                {
                    "role": "assistant",
                    "content": (
                        "Technical knowledge should be made as accessible as possible, with clear explanations and practical examples."
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How should I use you for study?"},
                {
                    "role": "assistant",
                    "content": (
                        "Ask short concrete questions, request examples, and test your understanding by asking for comparisons and exercises."
                    ),
                },
            ]
        },
    ]


SPELLING_WORDS = [
    "apple",
    "banana",
    "orange",
    "grapes",
    "strawberry",
    "blueberry",
    "raspberry",
    "computer",
    "algorithm",
    "dataset",
    "gradient",
    "attention",
    "embedding",
    "transformer",
    "language",
    "network",
    "training",
    "inference",
    "python",
    "science",
    "lecture",
    "student",
    "teacher",
    "research",
    "mathematics",
    "probability",
    "statistics",
    "sequence",
    "context",
    "generation",
    "learning",
    "optimization",
]


def _build_simple_spelling_rows(size: int) -> list[dict[str, object]]:
    templates = [
        "Spell the word '{word}'.",
        "How do you spell '{word}'?",
        "Write the correct spelling of '{word}'.",
    ]
    rows: list[dict[str, object]] = []
    for index in range(max(0, size)):
        word = SPELLING_WORDS[index % len(SPELLING_WORDS)]
        prompt = templates[index % len(templates)].format(word=word)
        answer = f"The word is spelled {'-'.join(word.upper())}."
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return rows


def _build_spelling_bee_rows(size: int) -> list[dict[str, object]]:
    templates = [
        "How many '{char}' letters are in '{word}'?",
        "Count the number of '{char}' characters in '{word}'.",
        "How many times does '{char}' appear in '{word}'?",
    ]
    rows: list[dict[str, object]] = []
    for index in range(max(0, size)):
        word = SPELLING_WORDS[index % len(SPELLING_WORDS)]
        letters = sorted(set(character for character in word if character.isalpha()))
        char = letters[index % len(letters)]
        count = word.count(char)
        prompt = templates[index % len(templates)].format(char=char, word=word)
        noun = "time" if count == 1 else "times"
        answer = f"The letter '{char}' appears {count} {noun} in '{word}'."
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return rows


def _gsm8k_to_messages(example: dict[str, object]) -> dict[str, object]:
    question = str(example["question"]).strip()
    answer = str(example["answer"]).strip()
    return {
        "messages": [
            {"role": "user", "content": f"Solve this math word problem.\n\n{question}"},
            {"role": "assistant", "content": answer},
        ]
    }


def _mmlu_to_messages(example: dict[str, object]) -> dict[str, object]:
    choices = [str(choice).strip() for choice in example["choices"]]  # type: ignore[index]
    letters = _choice_letters(len(choices))
    answer_index = int(example["answer"])
    choice_lines = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=False))
    answer_letter = letters[answer_index]
    answer_text = choices[answer_index]
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Answer this multiple-choice question.\n\n{example['question']}\n\nChoices:\n{choice_lines}",
            },
            {
                "role": "assistant",
                "content": f"The correct answer is {answer_letter}. {answer_text}",
            },
        ]
    }


def load_nanochat_like_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int | None = None,
    smoltalk_epochs: int = 1,
    everyday_epochs: int = 1,
    identity_epochs: int = 2,
    mmlu_epochs: int = 1,
    gsm8k_epochs: int = 1,
    simple_spelling_size: int = 20000,
    spelling_bee_size: int = 8000,
    seed: int = 42,
) -> Dataset:
    pieces: list[Dataset] = []

    if smoltalk_epochs > 0:
        smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
        smoltalk = _tokenize_messages_dataset(smoltalk, tokenizer, max_length=max_length)
        pieces.append(_repeat_dataset(smoltalk, smoltalk_epochs))

    if everyday_epochs > 0:
        everyday = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
        everyday = _tokenize_messages_dataset(everyday, tokenizer, max_length=max_length)
        pieces.append(_repeat_dataset(everyday, everyday_epochs))

    if identity_epochs > 0:
        identity_rows = _build_identity_conversations()
        identity = Dataset.from_list(identity_rows)
        identity = _tokenize_messages_dataset(identity, tokenizer, max_length=max_length)
        pieces.append(_repeat_dataset(identity, identity_epochs))

    if mmlu_epochs > 0:
        mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        mmlu = mmlu.map(_mmlu_to_messages, remove_columns=mmlu.column_names)
        mmlu = _tokenize_messages_dataset(mmlu, tokenizer, max_length=max_length)
        pieces.append(_repeat_dataset(mmlu, mmlu_epochs))

    if gsm8k_epochs > 0:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        gsm8k = gsm8k.map(_gsm8k_to_messages, remove_columns=gsm8k.column_names)
        gsm8k = _tokenize_messages_dataset(gsm8k, tokenizer, max_length=max_length)
        pieces.append(_repeat_dataset(gsm8k, gsm8k_epochs))

    if simple_spelling_size > 0:
        simple_spelling = Dataset.from_list(_build_simple_spelling_rows(simple_spelling_size))
        simple_spelling = _tokenize_messages_dataset(simple_spelling, tokenizer, max_length=max_length)
        pieces.append(simple_spelling)

    if spelling_bee_size > 0:
        spelling_bee = Dataset.from_list(_build_spelling_bee_rows(spelling_bee_size))
        spelling_bee = _tokenize_messages_dataset(spelling_bee, tokenizer, max_length=max_length)
        pieces.append(spelling_bee)

    if not pieces:
        raise ValueError("Nanochat-like SFT mixture produced no datasets.")

    mixed = concatenate_datasets(pieces)
    return mixed.shuffle(seed=seed)


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
