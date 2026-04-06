from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy

from course_tools import DEFAULT_CHAT_MESSAGES, build_demo_bundle, format_messages, generate_text, train_model


if __name__ == '__main__':
    formatted = format_messages(DEFAULT_CHAT_MESSAGES + [{'role': 'assistant', 'content': 'Self-attention lets tokens mix information from earlier tokens.'}])
    print('formatted conversation:')
    print(formatted)

    bundle = build_demo_bundle(steps=20)
    base_model = bundle['model']
    tokenizer = bundle['tokenizer']
    prompt = format_messages(DEFAULT_CHAT_MESSAGES, add_assistant_prompt=True)
    base_reply = generate_text(base_model, tokenizer, prompt, max_new_tokens=40, temperature=0.0)
    print('\nbase reply:')
    print(repr(base_reply))

    sft_model = copy.deepcopy(base_model)
    sft_text = '\\n'.join([
        format_messages(DEFAULT_CHAT_MESSAGES + [{'role': 'assistant', 'content': 'Self-attention builds a weighted summary of earlier tokens.'}]),
        format_messages([
            {'role': 'system', 'content': 'You are concise.'},
            {'role': 'user', 'content': 'What is tokenization?'},
            {'role': 'assistant', 'content': 'Tokenization maps raw text into discrete IDs.'},
        ]),
    ] * 6)
    train_model(sft_model, tokenizer, train_text=sft_text, eval_text=sft_text, steps=10, learning_rate=2e-3, batch_size=4)
    sft_reply = generate_text(sft_model, tokenizer, prompt, max_new_tokens=40, temperature=0.0)
    print('\nSFT reply:')
    print(repr(sft_reply))
