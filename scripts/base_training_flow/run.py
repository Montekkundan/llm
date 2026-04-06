from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import build_demo_bundle, default_artifact_dir, save_checkpoint, write_json


if __name__ == '__main__':
    bundle = build_demo_bundle(steps=50)
    out_dir = default_artifact_dir() / 'base_training'
    checkpoint = save_checkpoint(out_dir / 'base_checkpoint.pt', bundle['model'], bundle['tokenizer'], bundle['config'], metadata=bundle['metadata'], history=bundle['history'])
    report = write_json(
        out_dir / 'train_report.json',
        {
            'checkpoint': str(checkpoint),
            'history': bundle['history'],
            'metadata': bundle['metadata'],
            'config': bundle['config'].__dict__,
        },
    )
    print('base checkpoint:', checkpoint)
    print('report         :', report)
