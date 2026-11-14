"""Put experiments/gnn on sys.path so the tests can `import gnn_embed` without packaging the experiments folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
