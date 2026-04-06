from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

from course_tools import DEFAULT_CHAT_MESSAGES, ensure_demo_checkpoint, format_messages, generate_text, prefill_prompt, decode_next_token


if __name__ == '__main__':
    bundle = ensure_demo_checkpoint(steps=40)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    prompt = format_messages(DEFAULT_CHAT_MESSAGES, add_assistant_prompt=True)

    prompt_ids, past_kvs = prefill_prompt(model, tokenizer, prompt)
    print('prefill tokens:', len(prompt_ids))
    print('layer cache shapes:', [kv[0].shape for kv in past_kvs])

    current = prompt_ids[-1]
    for step in range(3):
        next_id, past_kvs, _ = decode_next_token(model, current, past_kvs, temperature=0.8, top_k=8)
        print({'step': step, 'token_id': next_id, 'token': tokenizer.decode([next_id]), 'cache_len': past_kvs[0][0].size(-2)})
        current = next_id

    start = time.perf_counter()
    generate_text(model, tokenizer, prompt, max_new_tokens=32, temperature=0.8, top_k=8)
    cached_time = time.perf_counter() - start

    start = time.perf_counter()
    text = prompt
    for _ in range(32):
        text += generate_text(model, tokenizer, text, max_new_tokens=1, temperature=0.8, top_k=8)
    naive_time = time.perf_counter() - start
    print('cached generation time:', round(cached_time, 4))
    print('naive recompute time  :', round(naive_time, 4))
