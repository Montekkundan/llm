from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import DEFAULT_CHAT_MESSAGES, ensure_demo_checkpoint, format_messages, generate_text


if __name__ == '__main__':
    bundle = ensure_demo_checkpoint(steps=40)
    prompt = format_messages(DEFAULT_CHAT_MESSAGES, add_assistant_prompt=True)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    settings = [
        ('greedy', 0.0, None),
        ('cool', 0.4, 4),
        ('balanced', 0.8, 8),
        ('hotter', 1.2, 12),
    ]
    for label, temperature, top_k in settings:
        print('=' * 80)
        print(label, {'temperature': temperature, 'top_k': top_k})
        print(repr(generate_text(model, tokenizer, prompt, max_new_tokens=60, temperature=temperature, top_k=top_k)))
