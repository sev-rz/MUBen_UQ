"""Summaries of the conformal results under PROTOCOL_v2.md section 3 (rules fixed before the full run; reads the CSVs written by conformal_v2.py only).
  python conformal_summary.py [IN_DIR] [OUT_DIR]     defaults: results_v2 -> results_v2   (selftest_v2 can be used to test the code on validation halves)
Primary quantity at alpha 0.1: regression = mean width of the normalised-score interval, classification = mean LAC set size (tasks averaged by conformal_v2.py).
Pairing: same backbone, split, dataset and seed (scaffold seeds 0/100/200; random: one seed per split). Scaffold Ensembles are one pooled value per dataset.
Contrasts (delta = first minus second; negative = narrower / smaller): P1 Laplace - Deterministic, P2 Laplace - Ensembles, P3 Novelty Laplace - Laplace constant (scaffold only).
A dataset has a consistent sign only if all repeats agree. A case 'holds coverage' if more than half of its tasks have verdict 'held'. A dataset is interpretable only if both
methods hold coverage in every repeat. An overall statement needs the same consistent sign on at least 6 of 8 interpretable datasets for that split type and backbone."""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
C = Path(__file__).resolve().parent
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else C / 'results_v2'; OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else IN
A = 0.1; KEYS = ['base', 'stype', 'split', 'dataset', 'seed']
CONTRASTS = [('P1', 'Laplace', 'Deterministic', None), ('P2', 'Laplace', 'Ensembles', None), ('P3', 'Novelty Laplace', 'Laplace constant', 'scaffold')]

def wilson(x, n, z=1.96):
    if n <= 0 or not np.isfinite(n): return (np.nan, np.nan)
    p = x / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h

def prep(kind):
    f = IN / ('conformal_regression.csv' if kind == 'reg' else 'conformal_classification.csv')
    df = pd.read_csv(f); df = df[(df.alpha == A) & ~df.backbone.str.endswith('-v2')].copy()
    df['base'] = df.backbone.str.replace('-Laplace', '', regex=False); df['stype'] = np.where(df.split == 'scaffold', 'scaffold', 'random'); df['seed'] = df.seed.astype(str)
    df = df[df.score == ('normalised' if kind == 'reg' else 'LAC')].copy(); df['value'] = df.width_mean if kind == 'reg' else df.set_size
    df['holds'] = df.verdict_held_tasks * 2 > df.tasks; return df

def pair(df, a, b, only):
    x = df[df.method == a][KEYS + ['value', 'coverage', 'holds']]; y = df[df.method == b][KEYS + ['value', 'coverage', 'holds']]
    if only: x = x[x.stype == only]; y = y[y.stype == only]
    if (y.seed == 'pooled').any():
        y = y.drop(columns='seed'); m = x.merge(y, on=[k for k in KEYS if k != 'seed'], suffixes=('', '_b'))
    else: m = x.merge(y, on=KEYS, suffixes=('', '_b'))
    m['delta'] = m.value - m.value_b; m['both_hold'] = m.holds & m.holds_b; return m

def consistency(s): return 'narrower' if (s < 0).all() else ('wider' if (s > 0).all() else 'mixed')

def contrasts(df, kind):
    rows, tally = [], []
    for name, a, b, only in CONTRASTS:
        m = pair(df, a, b, only)
        if m.empty: continue
        for (base, st, d), g in m.groupby(['base', 'stype', 'dataset']):
            full = len(g) >= (3 if st == 'scaffold' else 3)
            rows.append({'kind': kind, 'contrast': f'{name}: {a} - {b}', 'base': base, 'stype': st, 'dataset': d, 'repeats': len(g), 'mean_delta': g.delta.mean(), 'sd_delta': g.delta.std(ddof=0),
                         'sign_all': consistency(g.delta), 'cov_a': g.coverage.mean(), 'cov_b': g.coverage_b.mean(), 'both_hold_all': bool(g.both_hold.all()) and full,
                         'sign_interpretable': consistency(g.delta) if (g.both_hold.all() and full) else 'not interpretable'})
    r = pd.DataFrame(rows)
    if r.empty: return r, r
    t = r.groupby(['kind', 'contrast', 'base', 'stype']).agg(datasets=('dataset', 'size'), narrower_all=('sign_all', lambda s: (s == 'narrower').sum()), wider_all=('sign_all', lambda s: (s == 'wider').sum()),
                                                         mixed_all=('sign_all', lambda s: (s == 'mixed').sum()), interpretable=('both_hold_all', 'sum'),
                                                         narrower_interp=('sign_interpretable', lambda s: (s == 'narrower').sum()), wider_interp=('sign_interpretable', lambda s: (s == 'wider').sum())).reset_index()
    t['overall_statement'] = np.where(t.narrower_interp >= 6, 'narrower (>=6 of 8 interpretable)', np.where(t.wider_interp >= 6, 'wider (>=6 of 8 interpretable)', 'no overall statement'))
    return r, t

def coverage_shares(df, kind):
    """Q2: share of cases whose coverage verdict is 'held' (overall, and by novelty subgroup with at least 20 test molecules, Wilson interval around the subgroup coverage)."""
    d = df.copy(); d['held_all'] = d.holds
    for g in ('novel', 'familiar'):
        n = d[f'n_novel_test'] if g == 'novel' else d.n_test - d.n_novel_test; c = d[f'cov_{g}']
        ok = (n >= 20) & c.notna(); lo_hi = [wilson(round(ci * ni), ni) if o else (np.nan, np.nan) for ci, ni, o in zip(c, n, ok)]
        d[f'held_{g}'] = [(lo <= 1 - A <= hi) if o else np.nan for (lo, hi), o in zip(lo_hi, ok)]
    return d.groupby(['base', 'method', 'stype']).agg(cases=('held_all', 'size'), held_all=('held_all', 'mean'), held_novel=('held_novel', 'mean'), held_familiar=('held_familiar', 'mean'),
                                                        cov_all=('coverage', 'mean'), cov_novel=('cov_novel', 'mean'), cov_familiar=('cov_familiar', 'mean')).reset_index().round(3)

def mondrian(df, kind):
    """Q3: novel-subgroup coverage under marginal vs Mondrian calibration, and the cost: regression = Mondrian novel-subgroup width / overall marginal width - 1 (relative, so datasets of different scale can be pooled); classification = Mondrian novel-subgroup set size - overall marginal set size. Not like-for-like (the marginal width on the novel subgroup alone is not stored), so read it as an indication only."""
    cost = ('mondrian_width_novel', 'width_mean') if kind == 'reg' else ('mondrian_size_novel', 'set_size')
    d = df[df.mondrian_cov_novel.notna()].copy(); d['cov_marginal_novel'] = d.cov_novel; d['cost'] = (d[cost[0]] / d[cost[1]] - 1) if kind == 'reg' else (d[cost[0]] - d[cost[1]])
    d['gap_marginal'] = (d.cov_novel - (1 - A)).abs(); d['gap_mondrian'] = (d.mondrian_cov_novel - (1 - A)).abs()
    return d.groupby(['base', 'method', 'stype']).agg(cases=('cost', 'size'), cov_novel_marginal=('cov_novel', 'mean'), cov_novel_mondrian=('mondrian_cov_novel', 'mean'),
                                                      gap_marginal=('gap_marginal', 'mean'), gap_mondrian=('gap_mondrian', 'mean'), median_cost=('cost', 'median')).reset_index().round(3)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True); pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    for kind in ('reg', 'cls'):
        df = prep(kind)
        if df.empty: print(kind, 'no rows'); continue
        r, t = contrasts(df, kind); q2 = coverage_shares(df, kind); q3 = mondrian(df, kind)
        r.to_csv(OUT / f'summary_contrasts_{kind}.csv', index=False); t.to_csv(OUT / f'summary_tally_{kind}.csv', index=False)
        q2.to_csv(OUT / f'summary_q2_coverage_{kind}.csv', index=False); q3.to_csv(OUT / f'summary_q3_mondrian_{kind}.csv', index=False)
        print(f'\n=== {kind}: tally of contrasts (datasets of 8)'); print(t.to_string(index=False))
        print(f'--- {kind}: Q2 coverage verdict shares (Deterministic, Ensembles, Laplace only)'); print(q2[q2.method.isin(['Deterministic', 'Ensembles', 'Laplace'])].to_string(index=False))
        print(f'--- {kind}: Q3 Mondrian vs marginal on the novel subgroup (Deterministic, Laplace only)'); print(q3[q3.method.isin(['Deterministic', 'Laplace'])].to_string(index=False))
