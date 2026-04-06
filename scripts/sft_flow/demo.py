from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy

from course_tools import build_demo_bundle, default_artifact_dir, format_messages, generate_text, save_checkpoint, train_model, write_json


if __name__ == '__main__':
    bundle = build_demo_bundle(steps=30)
    base_model = bundle['model']
    tokenizer = bundle['tokenizer']
    prompt_messages = [
        {'role': 'system', 'content': 'You are concise.'},
        {'role': 'user', 'content': 'What is self-attention?'},
    ]
    prompt = format_messages(prompt_messages, add_assistant_prompt=True)
    base_text = generate_text(base_model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)

    sft_model = copy.deepcopy(base_model)
    sft_corpus = '\\n'.join([
        format_messages(prompt_messages + [{'role': 'assistant', 'content': 'Self-attention lets each token build a weighted summary of earlier tokens.'}]),
        format_messages([
            {'role': 'system', 'content': 'You are concise.'},
            {'role': 'user', 'content': 'What is an embedding layer?'},
            {'role': 'assistant', 'content': 'It maps token IDs to learned vectors.'},
        ]),
    ] * 8)
    history = train_model(sft_model, tokenizer, train_text=sft_corpus, eval_text=sft_corpus, steps=15, learning_rate=2e-3, batch_size=4)
    sft_text = generate_text(sft_model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)

    out_dir = default_artifact_dir() / 'sft_flow'
    checkpoint = save_checkpoint(out_dir / 'sft_checkpoint.pt', sft_model, tokenizer, bundle['config'], metadata={'stage': 'sft'}, history=history)
    report = write_json(out_dir / 'sft_report.json', {'base_text': base_text, 'sft_text': sft_text, 'checkpoint': str(checkpoint), 'history': history})
    print('base:', repr(base_text))
    print('sft :', repr(sft_text))
    print('checkpoint:', checkpoint)
    print('report    :', report)
