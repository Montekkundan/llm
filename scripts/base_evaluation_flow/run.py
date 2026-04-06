from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import DEFAULT_CHAT_MESSAGES, available_checkpoints, default_artifact_dir, ensure_demo_checkpoint, evaluate_model, format_messages, generate_text, write_json


if __name__ == '__main__':
    bundle = ensure_demo_checkpoint(steps=40)
    eval_text = '\\n'.join([
        'validation loss and bits per byte help us compare checkpoints honestly.',
        'sample generations give qualitative context beside the metrics.',
    ] * 12)
    metrics = evaluate_model(bundle['model'], bundle['tokenizer'], eval_text)
    sample = generate_text(
        bundle['model'],
        bundle['tokenizer'],
        format_messages(DEFAULT_CHAT_MESSAGES, add_assistant_prompt=True),
        max_new_tokens=60,
        temperature=0.8,
        top_k=8,
    )
    out = write_json(
        default_artifact_dir() / 'base_evaluation' / 'eval_report.json',
        {
            'metrics': metrics,
            'sample_generation': sample,
            'available_checkpoints': available_checkpoints(),
        },
    )
    print('metrics:', metrics)
    print('report :', out)
