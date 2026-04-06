from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import ensure_demo_checkpoint, format_messages, generate_text


def main() -> None:
    bundle = ensure_demo_checkpoint(steps=40)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    system = {'role': 'system', 'content': 'You are a concise teaching assistant for an LLM course.'}
    print("Type 'exit' to stop.")
    while True:
        user_text = input('user> ').strip()
        if user_text.lower() in {'exit', 'quit'}:
            break
        messages = [system, {'role': 'user', 'content': user_text}]
        prompt = format_messages(messages, add_assistant_prompt=True)
        reply = generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=8).strip() or 'The demo model produced an empty sample. Try a shorter prompt.'
        print('assistant>', reply)


if __name__ == '__main__':
    main()
