from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from course_tools import CharTokenizer, TinyConfig, TinyTransformerLM


if __name__ == '__main__':
    text = 'hello transformers'
    tokenizer = CharTokenizer.build([text])
    ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True))
    x = ids[:-1][None, :]
    y = ids[1:][None, :]
    model = TinyTransformerLM(TinyConfig(vocab_size=len(tokenizer.stoi), block_size=32, d_model=32, n_heads=4, n_layers=1, d_ff=64))
    logits, loss, _ = model(x, targets=y)
    print('input ids :', x)
    print('target ids:', y)
    print('logits shape:', tuple(logits.shape))
    print('loss:', float(loss))
    next_token_logits = logits[0, -1]
    top = torch.topk(next_token_logits, k=5)
    print('top next-token ids:', top.indices.tolist())
    print('top next-token chars:', [tokenizer.decode([i]) for i in top.indices.tolist()])
