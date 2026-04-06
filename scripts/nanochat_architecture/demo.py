from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pprint import pprint

from course_tools import default_artifact_dir


if __name__ == '__main__':
    architecture = {
        'concept notebooks': 'notebooks/<topic>/lecture_walkthrough.ipynb',
        'concept demos': 'scripts/<topic>/demo.py',
        'shared runtime': 'course_tools/runtime.py',
        'training flow': 'scripts/base_training_flow/run.py',
        'evaluation flow': 'scripts/base_evaluation_flow/run.py',
        'inference': 'scripts/inference_and_sampling/demo.py',
        'chat serving': 'scripts/fastapi_chat_app/app.py',
        'deployment': ['Dockerfile', 'RUN_APP.md', 'scripts/deployment/'],
        'artifacts': str(default_artifact_dir()),
    }
    pprint(architecture)
