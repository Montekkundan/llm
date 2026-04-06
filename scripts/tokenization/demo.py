from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache


def word_level_tokenize(text: str) -> list[str]:
    return text.split()


def character_level_tokenize(text: str) -> list[str]:
    return list(text)


def rasbt_split(text: str) -> list[str]:
    pattern = r"([,.:;?_!\"()']|--|\\s)"
    pieces = re.split(pattern, text)
    return [item.strip() for item in pieces if item.strip()]


def get_pair_stats(vocab_with_counts: dict[str, int]) -> Counter:
    stats = Counter()
    for word, freq in vocab_with_counts.items():
        symbols = word.split()
        for pair in zip(symbols, symbols[1:]):
            stats[pair] += freq
    return stats


def sentencepiece_pretokenize(text: str) -> str:
    return "▁" + text.replace(" ", "▁")


def unigram_segment(text: str, scores: dict[str, float]) -> tuple[float, list[str]]:
    @lru_cache(maxsize=None)
    def best(i: int):
        if i == len(text):
            return 0.0, []
        best_score = -float("inf")
        best_path: list[str] = []
        for j in range(i + 1, len(text) + 1):
            piece = text[i:j]
            if piece not in scores:
                continue
            tail_score, tail_path = best(j)
            total = scores[piece] + tail_score
            if total > best_score:
                best_score = total
                best_path = [piece] + tail_path
        return best_score, best_path

    return best(0)


if __name__ == "__main__":
    print("word-level:", word_level_tokenize("hello world"))
    print("character-level:", character_level_tokenize("hello"))
    print("rasbt regex split:", rasbt_split("It's the last he painted, you know."))

    toy_vocab = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3,
    }
    print("top BPE pairs:", get_pair_stats(toy_vocab).most_common(5))

    text = sentencepiece_pretokenize("tokenization works")
    unigram_scores = {
        "▁tokenization": -0.05,
        "▁token": -0.2,
        "ization": -0.4,
        "▁works": -0.2,
    }
    print("SentencePiece string:", text)
    print("Best unigram path:", unigram_segment(text, unigram_scores))
