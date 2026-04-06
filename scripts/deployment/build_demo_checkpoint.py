from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_tools import ensure_demo_checkpoint


if __name__ == '__main__':
    bundle = ensure_demo_checkpoint(steps=40)
    print('checkpoint ready at:', bundle['path'])
