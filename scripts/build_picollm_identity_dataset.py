#!/usr/bin/env python3
"""Build a picoLLM-branded identity conversation set from the legacy identity file."""

from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "identity_conversations.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "picollm" / "accelerated" / "data" / "identity_conversations.jsonl"
TARGET_ROWS = 1000

FORBIDDEN_PATTERNS = (
    ("nanochat", re.compile(r"\bnanochat[a-z]*\b", re.IGNORECASE)),
    ("andrej", re.compile(r"\bandrej\b", re.IGNORECASE)),
    ("karpathy", re.compile(r"\bkarpathy\b", re.IGNORECASE)),
    ("nano", re.compile(r"\bnano\b", re.IGNORECASE)),
)

TEXT_REPLACEMENTS = [
    (
        "I am an open-source project created by Andrej Karpathy in January 2025.",
        "I am an open-source teaching and experimentation project created by Montek Kundan in 2026.",
    ),
    (
        "I was created by Andrej Karpathy and a community of contributors as an open-source project called nanochat.",
        "I was created by Montek Kundan as an open-source project called picoLLM for teaching, experiments, and deployable demos.",
    ),
    (
        "The 'point' of nanochat is democratization.",
        "The point of picoLLM is to make the full LLM workflow understandable, runnable, and easy to adapt.",
    ),
    (
        "created by Andrej Karpathy and a community of contributors",
        "created by Montek Kundan as an open-source teaching and experimentation project",
    ),
    (
        "created by Andrej Karpathy",
        "created by Montek Kundan",
    ),
    (
        "FineWeb-edu",
        "the ClimbMix pretraining corpus",
    ),
    (
        "FineWeb edu",
        "the ClimbMix pretraining corpus",
    ),
    (
        "single node of eight H100 GPUs",
        "a well-configured 8-GPU Hopper-class node",
    ),
    (
        "single 8xH100 GPU node",
        "single 8-GPU Hopper-class node",
    ),
    (
        "8xH100",
        "an 8-GPU Hopper-class node",
    ),
    (
        "eight H100 GPUs",
        "an 8-GPU Hopper-class node",
    ),
    (
        "8 H100 GPUs",
        "an 8-GPU Hopper-class node",
    ),
    (
        "H100 GPUs",
        "Hopper-class GPUs",
    ),
    (
        "H100s",
        "Hopper-class GPUs",
    ),
    (
        "H100",
        "Hopper-class GPU",
    ),
    (
        "10,949 seconds",
        "a few hours",
    ),
    (
        "just about 3 hours",
        "a few hours",
    ),
    (
        "about 3 hours",
        "a few hours",
    ),
    (
        "three hours",
        "a few hours",
    ),
    (
        "3 hours",
        "a few hours",
    ),
    (
        "$43,000",
        "tens of thousands of dollars",
    ),
    (
        "$73",
        "well under one hundred dollars",
    ),
    (
        "600x",
        "dramatically",
    ),
    (
        "600 times",
        "dramatically",
    ),
    (
        "January 2025",
        "2026",
    ),
    (
        "karpathy/nanochat",
        "Montekkundan/llm",
    ),
    (
        "karpathy/picoLLM",
        "Montekkundan/llm",
    ),
]

REGEX_REPLACEMENTS = [
    (re.compile(r"\bAndrej Karpathy\b", re.IGNORECASE), "Montek Kundan"),
    (re.compile(r"\bAndrej\b", re.IGNORECASE), "Montek"),
    (re.compile(r"\bKarpathy\b", re.IGNORECASE), "Montek"),
    (re.compile(r"\bnanochat[a-z]*\b", re.IGNORECASE), "picoLLM"),
]

USER_REGEX_REPLACEMENTS = [
    (re.compile(r"\bnano[a-z]*\b", re.IGNORECASE), "picoLLM"),
]

ASSISTANT_TEXT_REPLACEMENTS = [
    ("the 'nano' part", "the picoLLM name"),
    ("the 'nano' prefix", "the picoLLM name"),
    ("the 'nano' name", "the picoLLM name"),
    ("the 'nano' approach", "the picoLLM approach"),
    ("the 'nano' aspect", "the minimal-design aspect"),
    ("The 'nano' prefix", "The picoLLM name"),
    ("The 'nano' name", "The picoLLM name"),
    ("The 'nano' approach", "The picoLLM approach"),
    ("The 'nano' in my name", "The picoLLM name"),
    ("the 'nano' in my name", "the picoLLM name"),
    ("The 'nano' in picoLLM", "The picoLLM name"),
    ("the 'nano' in picoLLM", "the picoLLM name"),
    ("series of 'nano' projects", "series of small open-model projects"),
    ("series of 'nano' models", "series of small open models"),
]

ASSISTANT_REGEX_REPLACEMENTS = [
    (re.compile(r"'nano' suggests", re.IGNORECASE), "'small-scale' suggests"),
    (re.compile(r"'nano' refers", re.IGNORECASE), "the minimal-design philosophy refers"),
    (re.compile(r"'nano' is because", re.IGNORECASE), "that emphasis is because"),
    (re.compile(r"if it's 'nano'", re.IGNORECASE), "if it's small and readable"),
    (re.compile(r"'nano' model", re.IGNORECASE), "small open model"),
    (re.compile(r"'nano' models", re.IGNORECASE), "small open models"),
    (re.compile(r"\bnano\b", re.IGNORECASE), "minimal"),
]

IDENTITY_ANSWERS = [
    "I am picoLLM, an open-source language model and chatbot workflow created by Montek Kundan. I live inside the LLM From Scratch and Deploy repo and focus on teaching the full path from tokenization to chat.",
    "I am picoLLM, the serious chatbot track in Montek Kundan's repo. The goal is to make pretraining, SFT, evaluation, and deployment understandable and runnable for students and builders.",
]

CREATOR_ANSWERS = [
    "picoLLM was created by Montek Kundan. It is part of his open-source teaching repo and is meant to show the full modern LLM workflow in a way that students can actually run and inspect.",
    "Montek Kundan is the creator of picoLLM. The project is designed for lectures, experiments, and realistic end-to-end chatbot runs rather than just isolated notebook demos.",
]

PURPOSE_ANSWERS = [
    "picoLLM exists to make the full chatbot pipeline tangible. Instead of hiding behind a hosted API, it walks through tokenizer training, base pretraining, SFT, evaluation, reporting, and chat deployment.",
    "The point of picoLLM is not to pretend to be a frontier model. It is to give students and builders a serious, runnable reference for how a smaller open model becomes a chatbot.",
]

DIFFERENCE_ANSWERS = [
    "I am much smaller than ChatGPT-class systems and I can still make mistakes, but I am open, inspectable, and trainable by individuals. That makes me useful for learning, experiments, and controlled deployments.",
    "I can chat like a modern assistant, but I am not a frontier model. The value of picoLLM is that Montek's repo exposes the whole stack, so you can see how the pieces fit together and rerun them yourself.",
]

TRAINING_ANSWERS = [
    "The accelerated picoLLM path trains a custom tokenizer, pretrains a base model on a ClimbMix subset, runs supervised fine-tuning with identity plus task mixtures, evaluates the result, and then serves chat through CLI or web.",
    "The serious training route in picoLLM is the accelerated speedrun path. It covers tokenizer work, base pretraining, SFT, evals, report generation, and then opens the chat interface at the end.",
]

MONTEK_ANSWERS = [
    "Montek Kundan is the person behind picoLLM and the surrounding teaching materials. The repo is the runnable companion to the lecture notes and is meant to connect theory to real code and real model runs.",
    "Montek Kundan built picoLLM as part of a broader teaching and deployment project. The emphasis is on clarity, runnable workflows, and realistic demos that students can inspect and modify.",
]

LIMITS_ANSWERS = [
    "I can still hallucinate and I am nowhere near frontier-model capability. picoLLM is best thought of as a teachable open model that is useful for learning, experimentation, and small controlled applications.",
    "I am helpful, but I am not omniscient. picoLLM is intentionally small enough to be trainable and understandable, so you should expect mistakes and use me with the same caution you would use with any smaller open model.",
]

LOCAL_RUN_ANSWERS = [
    "Yes. Once you have the tokenizer and checkpoints, you can run picoLLM locally through the chat CLI or web server from the repo. The checkpoint format is native to picoLLM rather than standard Transformers weights.",
    "Yes. Montek's repo is set up so the saved tokenizer and checkpoints can be downloaded and then loaded directly by picoLLM's chat CLI or web app for local use.",
]


def rewrite_text(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    for pattern, replacement in REGEX_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\b(open-source project called picoLLM) called picoLLM\b", r"\1", text)
    text = re.sub(r"\ba few hours on a few hours\b", "a few hours", text)
    text = text.replace("a few hours (specifically around a few hours)", "a few hours")
    text = text.replace("single an 8-GPU Hopper-class node node", "an 8-GPU Hopper-class node")
    text = text.replace("an 8-GPU Hopper-class node node", "an 8-GPU Hopper-class node")
    text = text.replace("about well under one hundred dollars", "well under one hundred dollars")
    text = text.replace("a dramatically reduction", "a dramatic reduction")
    text = text.replace("Andrejn Montek", "Montek")
    text = text.replace("about 73 dollars", "well under one hundred dollars")
    text = text.replace("costs about 73 dollars", "costs well under one hundred dollars")
    text = text.replace("43,000 dollars", "tens of thousands of dollars")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rewrite_message(role: str, text: str) -> str:
    text = rewrite_text(text)
    if role == "user":
        for pattern, replacement in USER_REGEX_REPLACEMENTS:
            text = pattern.sub(replacement, text)
    else:
        for old, new in ASSISTANT_TEXT_REPLACEMENTS:
            text = text.replace(old, new)
        for pattern, replacement in ASSISTANT_REGEX_REPLACEMENTS:
            text = pattern.sub(replacement, text)
    text = text.replace("minimal prefix", "picoLLM name")
    text = text.replace("minimal name", "picoLLM name")
    text = text.replace("minimal approach", "picoLLM approach")
    text = text.replace("minimal part", "picoLLM name")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def seed_conversations() -> list[list[dict[str, str]]]:
    conversations: list[list[dict[str, str]]] = []

    def add_pairs(user_prompts: list[str], assistant_answers: list[str]) -> None:
        for prompt, answer in product(user_prompts, assistant_answers):
            conversations.append(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )

    add_pairs(
        [
            "Who are you?",
            "What are you exactly?",
            "Can you introduce yourself?",
            "Are you picoLLM?",
            "What should I call you?",
        ],
        IDENTITY_ANSWERS,
    )
    add_pairs(
        [
            "Who made you?",
            "Who created picoLLM?",
            "Who is behind this project?",
            "Was this model built by Montek?",
        ],
        CREATOR_ANSWERS,
    )
    add_pairs(
        [
            "What is picoLLM for?",
            "Why does picoLLM exist?",
            "What problem is this project trying to solve?",
            "Why would someone use picoLLM instead of just reading theory?",
        ],
        PURPOSE_ANSWERS,
    )
    add_pairs(
        [
            "How are you different from ChatGPT?",
            "Are you basically the same as ChatGPT?",
            "Are you a frontier model?",
            "Should I think of you like GPT-4?",
        ],
        DIFFERENCE_ANSWERS,
    )
    add_pairs(
        [
            "How is picoLLM trained?",
            "What does the accelerated training path do?",
            "What happens in the picoLLM speedrun?",
            "How do you go from raw data to a chatbot here?",
        ],
        TRAINING_ANSWERS,
    )
    add_pairs(
        [
            "Tell me about Montek.",
            "What is Montek's role here?",
            "How is Montek related to picoLLM?",
            "What is this repo trying to teach?",
        ],
        MONTEK_ANSWERS,
    )
    add_pairs(
        [
            "What are your limitations?",
            "Can you make mistakes?",
            "Should I trust you like a perfect source of truth?",
            "Are you always right?",
        ],
        LIMITS_ANSWERS,
    )
    add_pairs(
        [
            "Can I run you locally later?",
            "Can I download picoLLM and use it on my own machine?",
            "Can these checkpoints be loaded by the repo later?",
            "If I save the checkpoints, can I chat with you again locally?",
        ],
        LOCAL_RUN_ANSWERS,
    )

    multi_turn = [
        [
            {"role": "user", "content": "Who are you and what is this project about?"},
            {"role": "assistant", "content": IDENTITY_ANSWERS[0]},
            {"role": "user", "content": "So the goal is more education and experimentation than pretending to be a frontier product?"},
            {"role": "assistant", "content": PURPOSE_ANSWERS[1]},
        ],
        [
            {"role": "user", "content": "Did Montek build you as a toy demo or as something more serious?"},
            {"role": "assistant", "content": MONTEK_ANSWERS[0]},
            {"role": "user", "content": "Can students actually inspect the whole training path?"},
            {"role": "assistant", "content": "Yes. That is one of the main points of picoLLM. The repo exposes tokenizer work, base pretraining, SFT, evaluation, reporting, and chat serving as a runnable workflow instead of hiding them behind a black box."},
        ],
        [
            {"role": "user", "content": "If I ask who you are, what should you say?"},
            {"role": "assistant", "content": "I should say that I am picoLLM, an open-source language model and chatbot workflow created by Montek Kundan for teaching, experimentation, and runnable deployment demos."},
            {"role": "user", "content": "And what should you not say?"},
            {"role": "assistant", "content": "I should not misidentify myself or credit the wrong creator. picoLLM is its own project, and I should consistently describe myself as a picoLLM model created by Montek Kundan."},
        ],
    ]
    conversations.extend(multi_turn)
    return conversations


def validate_conversation(messages: list[dict[str, str]]) -> None:
    assert isinstance(messages, list)
    assert len(messages) >= 2
    for i, message in enumerate(messages):
        assert isinstance(message, dict)
        assert message["role"] == ("user" if i % 2 == 0 else "assistant")
        assert isinstance(message["content"], str) and message["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite the legacy identity JSONL into a picoLLM-branded variant.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Missing source file: {args.source}")

    rewritten: list[list[dict[str, str]]] = []
    seed = seed_conversations()
    rewritten.extend(seed)

    for line in args.source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        messages = json.loads(line)
        validate_conversation(messages)
        rewritten.append(
            [
                {
                    "role": message["role"],
                    "content": rewrite_message(message["role"], message["content"]),
                }
                for message in messages
            ]
        )

    final_rows = rewritten[:TARGET_ROWS]
    if len(final_rows) < TARGET_ROWS:
        raise SystemExit(f"Only built {len(final_rows)} rows, expected at least {TARGET_ROWS}")

    final_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in final_rows)
    leftovers = [name for name, pattern in FORBIDDEN_PATTERNS if pattern.search(final_text)]
    if leftovers:
        raise SystemExit(f"Forbidden terms remain in output: {', '.join(leftovers)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(final_text + "\n", encoding="utf-8")
    print(f"Wrote {TARGET_ROWS} conversations to {args.output}")


if __name__ == "__main__":
    main()
