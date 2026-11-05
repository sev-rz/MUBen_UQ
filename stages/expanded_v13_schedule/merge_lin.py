import json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
for d in sys.argv[1:]:
    fs=[D/'metrics_seed'/f'{d}_{s}.json' for s in (0,100,200)]
    if all(f.exists() for f in fs):
        o=D/'metrics'/f'lin_det_{d}.json';o.parent.mkdir(exist_ok=True);o.write_text(json.dumps([json.loads(f.read_text()) for f in fs]));print('merged',d)
