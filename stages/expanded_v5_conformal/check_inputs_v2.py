"""Input coverage check for conformal_v2.py (run before the full run; reads file names only, no prediction values).
For every prepared validation folder preds/<backbone>/<split>/<dataset>/ (with DONE) it looks for the test counterpart exactly as conformal_v2.analyse does
(preds_test/, expanded_v4/results/, expanded_v8_gin/results/) and fails loudly if any is missing. Also checks that each expected backbone has 32 groups."""
import glob, sys
from pathlib import Path
C = Path(__file__).resolve().parent; V = C.parent / 'expanded_v4'; V8 = C.parent / 'expanded_v8_gin'
EXPECT = ['DNN', 'ChemBERTa', 'GIN', 'DNN-Laplace', 'ChemBERTa-Laplace', 'GIN-Laplace']
groups, files, missing = {}, {}, {}
for vdir in sorted(glob.glob(str(C / 'preds' / '*' / '*' / '*'))):
    p = Path(vdir)
    if not (p / 'DONE').exists(): continue
    bb, split, d = p.parts[-3:]; groups.setdefault(bb, set()).add((split, d))
    for vf in sorted(p.glob('*.npz')):
        files[bb] = files.get(bb, 0) + 1
        cands = [C / 'preds_test' / bb / split / d / vf.name, V / 'results' / bb / split / d / vf.name, V8 / 'results' / bb / split / d / vf.name]
        if not any(c.exists() for c in cands): missing.setdefault(bb, []).append(f'{split}/{d}/{vf.name}')
bad = False
for bb in sorted(set(EXPECT) | set(groups)):
    g = len(groups.get(bb, ())); m = len(missing.get(bb, ()))
    print(f'{bb:18s} groups {g:3d}/32  validation files {files.get(bb, 0):4d}  without test counterpart {m}' + ('' if bb in EXPECT else '  (not in primary scope)'))
    if bb in EXPECT and (g != 32 or m): bad = True
    if m: print('   e.g.', missing[bb][:4])
sys.exit(1 if bad else 0)
