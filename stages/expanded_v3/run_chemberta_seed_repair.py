"""Re-run one crashed ChemBERTa seed with the unchanged train_chemberta.fit (see OPERATIONS_LOG.md).

Per-seed processes skip the shared CHEMBERTA_FROZEN.json write, which the subsequent
all-seed `train_chemberta.py --stage fit` writes once; decision() already skips because
CHEMBERTA_DECISION.json exists. Training code, settings and checksums are untouched.
"""
import sys
from pathlib import Path
import train_chemberta as t
_save=t.e.save_json
t.e.save_json=lambda path,value:None if Path(path).name=='CHEMBERTA_FROZEN.json' else _save(path,value)
t.SEEDS=[int(sys.argv[1])];t.fit()
