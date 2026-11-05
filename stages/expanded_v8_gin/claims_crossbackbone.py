"""Cross-backbone paper claims, operational definitions registered in journal n0070 (before any GIN test result was looked at).

(a) lower-expressiveness models are better calibrated: on a dataset and split type the claim is supported if ChemBERTa has the largest
    Deterministic calibration error of the three backbones; supported overall on a split type if this holds on >= 6 of 8 datasets.
(b) the choice of UQ method matters more for larger models: gap = Deterministic calibration error - smallest mean calibration error over
    the published methods of that backbone (Laplace excluded); supported on a dataset if the ChemBERTa gap exceeds both the DNN and the GIN gap;
    supported overall if on >= 6 of 8 datasets.
Calibration error = ECE (15-bin, positive class) for classification, upstream regression CE for regression, as stored in per_seed_metrics.json.
Deterministic rows: mean over the three repeats (scaffold seeds 0/100/200; random-0, -1, -2). Robustness view: the same with NLL.
Descriptive counts, no tests. Usage: python claims_crossbackbone.py [--check-code]  (--check-code runs DNN + ChemBERTa only, writes nothing)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
V8 = Path(__file__).resolve().parent; V4 = V8.parent / 'expanded_v4'
DS = ['bace', 'bbbp', 'clintox', 'tox21', 'esol', 'freesolv', 'lipo', 'qm7']; REG = {'esol', 'freesolv', 'lipo', 'qm7'}
ROOTS = {'DNN': V4 / 'results' / 'DNN', 'ChemBERTa': V4 / 'results' / 'ChemBERTa', 'GIN': V8 / 'results' / 'GIN'}
SPLITS = {'scaffold': ['scaffold'], 'random': ['random-0', 'random-1', 'random-2']}

def load(backbone):
    rows = []
    for sp_type, sps in SPLITS.items():
        for sp in sps:
            for d in DS:
                f = ROOTS[backbone] / sp / d / 'per_seed_metrics.json'
                if not f.exists(): return None
                for r in json.loads(f.read_text()):
                    r['split_type'] = sp_type; r['repeat'] = sp if sp_type == 'random' else r['seed']; rows.append(r)
    return pd.DataFrame(rows)

def tables(frames, metric_cls, metric_reg):
    out = []
    for sp_type in SPLITS:
        for d in DS:
            col = metric_reg if d in REG else metric_cls; rec = {'split_type': sp_type, 'dataset': d, 'metric': col}
            gaps = {}
            for bb, df in frames.items():
                g = df[(df.split_type == sp_type) & (df.dataset == d)]
                per_method = g.groupby('method')[col].mean()
                det = g[g.method == 'Deterministic'].groupby('repeat')[col].mean()
                assert len(det) == 3, (bb, sp_type, d, len(det))
                rec[f'det_{bb}'] = float(det.mean()); rec[f'best_{bb}'] = float(per_method.min()); rec[f'best_method_{bb}'] = per_method.idxmin()
                gaps[bb] = rec[f'det_{bb}'] - rec[f'best_{bb}']; rec[f'gap_{bb}'] = gaps[bb]
            det_vals = {bb: rec[f'det_{bb}'] for bb in frames}
            rec['a_supported'] = bool(max(det_vals, key=det_vals.get) == 'ChemBERTa')
            rec['b_supported'] = bool(all(gaps['ChemBERTa'] > gaps[b] for b in frames if b != 'ChemBERTa'))
            out.append(rec)
    return pd.DataFrame(out)

def summary(t):
    return t.groupby('split_type').agg(datasets=('dataset', 'size'), a_supported=('a_supported', 'sum'), b_supported=('b_supported', 'sum')).reset_index()

if __name__ == '__main__':
    check = '--check-code' in sys.argv
    names = ['DNN', 'ChemBERTa'] if check else ['DNN', 'GIN', 'ChemBERTa']
    frames = {b: load(b) for b in names}
    missing = [b for b, f in frames.items() if f is None]
    assert not missing, f'results missing for {missing}'
    views = {'calibration (ECE / CE)': ('ece', 'ce'), 'NLL (robustness)': ('nll', 'nll')}
    res = {}
    for name, (mc, mr) in views.items():
        t = tables(frames, mc, mr); res[name] = t
        print(f'== {name}; backbones {names}'); print(summary(t).to_string(index=False))
        if not check: print(t.round(4).to_string(index=False))
    if not check:
        for name, t in res.items(): t.to_csv(V8 / 'results' / ('CLAIMS_CROSSBACKBONE_' + ('calibration' if name.startswith('cal') else 'nll') + '.csv'), index=False)
        (V8 / 'results' / 'CLAIMS_CROSSBACKBONE.json').write_text(json.dumps({n: summary(t).to_dict('records') for n, t in res.items()}, indent=1))
