from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import build_demo_bundle, default_artifact_dir, evaluate_model, save_checkpoint, write_json


if __name__ == '__main__':
    bundle = build_demo_bundle(steps=30)
    artifacts = default_artifact_dir() / 'training_loop_demo'
    checkpoint_path = save_checkpoint(
        artifacts / 'checkpoint.pt',
        model=bundle['model'],
        tokenizer=bundle['tokenizer'],
        config=bundle['config'],
        metadata=bundle['metadata'],
        history=bundle['history'],
    )
    metrics = evaluate_model(
        bundle['model'],
        bundle['tokenizer'],
        '\\n'.join(['training loops coordinate data, loss, gradients, updates, and metrics.'] * 8),
    )
    report_path = write_json(
        artifacts / 'report.json',
        {
            'checkpoint': str(checkpoint_path),
            'metrics': metrics,
            'history': bundle['history'],
        },
    )
    print('checkpoint:', checkpoint_path)
    print('report    :', report_path)
    print('last history row:', bundle['history'][-1])
