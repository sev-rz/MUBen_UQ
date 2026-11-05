"""Presentation tables, two figures and CONFORMAL_SUMMARY.md from the frozen run (PROTOCOL_v2.md section 6). Reads results_v2/ only; nothing is recomputed from predictions.
Not part of the frozen digest list (presentation only, written after the freeze). Colours: Okabe-Ito blue / orange / green for DNN / ChemBERTa / GIN, with marker shapes as second encoding.
  python conformal_figures.py        (cwd = expanded_v5_conformal; python with pandas + matplotlib)"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = Path(__file__).resolve().parent; R = C / 'results_v2'; FIG = R / 'figures'; FIG.mkdir(exist_ok=True)
A = 0.1; KEYS = ['base', 'stype', 'split', 'dataset', 'seed']
BB = ['DNN', 'ChemBERTa', 'GIN']; COL = {'DNN': '#0072B2', 'ChemBERTa': '#E69F00', 'GIN': '#009E73'}
BASE_M = ['Temperature', 'Focal Loss', 'MC dropout', 'SWAG', 'BBP', 'SGLD', 'Ensembles']; LAP_M = ['Laplace', 'Laplace constant', 'Novelty only', 'Novelty Laplace']
MK = {'Deterministic': 'o', 'Ensembles': 's', 'Laplace': '^', 'Laplace constant': '^', 'Novelty only': '^', 'Novelty Laplace': '^'}
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)

def load(kind):
    d = pd.read_csv(R / ('conformal_regression.csv' if kind == 'reg' else 'conformal_classification.csv')); d = d[(d.alpha == A) & ~d.backbone.str.endswith('-v2')].copy()
    d['base'] = d.backbone.str.replace('-Laplace', '', regex=False); d['stype'] = np.where(d.split == 'scaffold', 'scaffold', 'random'); d['seed'] = d.seed.astype(str)
    d = d[d.score == ('normalised' if kind == 'reg' else 'LAC')].copy(); d['value'] = d.width_mean if kind == 'reg' else d.set_size; d['holds'] = d.verdict_held_tasks * 2 > d.tasks
    return d

def ratios(d):
    det = d[d.method == 'Deterministic'][KEYS + ['value']].rename(columns={'value': 'v0'}); out = []
    for m in BASE_M + LAP_M:
        x = d[d.method == m]
        if x.empty: continue
        if (x.seed == 'pooled').any():
            y = det.groupby([k for k in KEYS if k != 'seed'], as_index=False).v0.mean(); z = x.merge(y, on=[k for k in KEYS if k != 'seed'])
        else: z = x.merge(det, on=KEYS)
        z = z[(z.value > 0) & (z.v0 > 0)].copy(); z['logr'] = np.log(z.value / z.v0); out.append(z)
    z = pd.concat(out)
    return z.groupby(['stype', 'base', 'method']).agg(cases=('logr', 'size'), ratio_to_deterministic=('logr', lambda s: float(np.exp(s.mean()))), coverage=('coverage', 'mean'), cov_det_note=('holds', 'mean')).reset_index().rename(columns={'cov_det_note': 'share_cases_holding'})

def fig_ratio(rt, kind, name, ylabel):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    methods = [m for m in BASE_M + LAP_M if m in rt.method.unique()]
    for a, st in zip(ax, ('scaffold', 'random')):
        w = 0.26
        for i, b in enumerate(BB):
            s = rt[(rt.stype == st) & (rt.base == b)].set_index('method').reindex(methods)
            a.bar(np.arange(len(methods)) + (i - 1) * w, s.ratio_to_deterministic - 1, width=w - 0.04, bottom=1, color=COL[b], label=b)
        top = max(1.25, min(1.45, 1 + 1.1 * float((rt.ratio_to_deterministic - 1).abs().clip(upper=0.25).max())))
        a.set_ylim(2 - top, top)
        for i, b in enumerate(BB):   # bars beyond the axis are clipped and labelled with their value
            s = rt[(rt.stype == st) & (rt.base == b)].set_index('method').reindex(methods)
            for j, v in enumerate(s.ratio_to_deterministic):
                if v == v and (v > top or v < 2 - top): a.text(j + (i - 1) * w, top - 0.01 if v > top else 2 - top + 0.01, f'{v:.1f}', ha='center', va='top' if v > top else 'bottom', fontsize=8)
        a.axhline(1, color='#444', lw=1); a.set_xticks(range(len(methods))); a.set_xticklabels(methods, rotation=40, ha='right', fontsize=9)
        a.set_title(f'{st} split', fontsize=11, loc='left'); a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', color='#ddd', lw=0.6); a.set_axisbelow(True)
    ax[0].set_ylabel(ylabel); ax[1].legend(frameon=False, fontsize=9, title='backbone', loc='upper right')
    fig.suptitle(name, x=0.01, ha='left', fontsize=12); fig.tight_layout(); fig.savefig(FIG / f'ratio_{kind}.png', dpi=140); plt.close(fig)

def fig_subgroups(ds):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8.5))
    for r, kind in enumerate(('reg', 'cls')):
        d = ds[kind]
        for c, st in enumerate(('scaffold', 'random')):
            a = ax[r, c]; g = d[d.stype == st].groupby(['base', 'method']).agg(fam=('cov_familiar', 'mean'), nov=('cov_novel', 'mean')).reset_index()
            for _, row in g.iterrows():
                if row.method not in MK and row.method not in BASE_M: continue
                a.scatter(row.fam, row.nov, s=34 if row.method in MK else 14, marker=MK.get(row.method, 'D'), color=COL[row.base], edgecolor='white', linewidth=0.8, zorder=3, alpha=1 if row.method in MK else 0.55)
            a.axhline(1 - A, color='#444', lw=0.8, ls='--'); a.axvline(1 - A, color='#444', lw=0.8, ls='--'); lo, hi = 0.65, 1.0; a.plot([lo, hi], [lo, hi], color='#bbb', lw=0.8)
            a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_title(f"{'regression' if kind == 'reg' else 'classification'}, {st} split", fontsize=11, loc='left'); a.spines[['top', 'right']].set_visible(False)
            a.set_xlabel('test coverage, familiar molecules'); a.set_ylabel('test coverage, novel molecules')
    h = [plt.Line2D([], [], marker='o', ls='', color=COL[b], label=b) for b in BB] + [plt.Line2D([], [], marker=m, ls='', color='#555', label=l) for l, m in (('Deterministic', 'o'), ('Ensembles', 's'), ('Laplace family', '^'), ('other published methods (faded)', 'D'))]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=9); fig.suptitle('Coverage on novel versus familiar test molecules at the 0.90 target (dashed); mean over datasets and repeats', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); fig.savefig(FIG / 'coverage_novel_vs_familiar.png', dpi=140); plt.close(fig)

if __name__ == '__main__':
    ds = {k: load(k) for k in ('reg', 'cls')}; tabs = {}
    for k, nm in (('reg', 'width (normalised score)'), ('cls', 'LAC set size')):
        rt = ratios(ds[k]); rt.round(4).to_csv(R / f'report_ratio_to_deterministic_{k}.csv', index=False); tabs[k] = rt
        fig_ratio(rt, k, f"{'Interval width' if k == 'reg' else 'Prediction-set size'} relative to the Deterministic network (geometric mean over datasets and repeats; 1 = same)", f'ratio ({nm})')
    fig_subgroups(ds)
    q5 = []
    for k in ('reg', 'cls'):
        d = ds[k][ds[k].method == 'Deterministic']
        q5.append(d.groupby(['stype', 'base']).agg(kind=('value', lambda s: k), cases=('coverage', 'size'), coverage=('coverage', 'mean'), cov_familiar=('cov_familiar', 'mean'), cov_novel=('cov_novel', 'mean'), share_holding=('holds', 'mean'), mean_value=('value', 'mean')).reset_index())
    q5 = pd.concat(q5).round(3); q5.to_csv(R / 'report_q5_deterministic.csv', index=False)
    q3r = pd.read_csv(R / 'summary_q3_mondrian_reg.csv'); q3c = pd.read_csv(R / 'summary_q3_mondrian_cls.csv')
    sel = lambda t: t[t.method.isin(['Deterministic', 'Ensembles', 'Laplace'])]
    sha = hashlib.sha256((C / 'PROTOCOL_v2.md').read_bytes()).hexdigest()
    md = ['# Conformal extension: summary of the frozen run (generated by conformal_figures.py; do not edit by hand)', '',
          f'Protocol: PROTOCOL_v2.md, sha256 {sha}. Predictions: saved validation (calibration) and test predictions of 3 backbones x (published methods + Laplace family); alpha {A}; 8 datasets, scaffold and random-0..2.',
          'Calibration set = the validation split that was also used for early stopping (and for alpha / lambda of the Laplace family), so the finite-sample guarantee is approximate. Test sets were opened earlier for the baselines: exploratory.', '',
          '## Q5: Deterministic network, coverage and size under scaffold shift versus random splits', '', '```', q5.to_string(index=False), '```', '',
          '## Interval width / set size relative to Deterministic (figures ratio_reg.png, ratio_cls.png)', '', '```',
          pd.concat([tabs['reg'].assign(kind='reg'), tabs['cls'].assign(kind='cls')]).round(3).to_string(index=False), '```', '',
          '## Q3: Mondrian calibration on the novel subgroup (regression: width cost relative; classification: set-size difference; not like-for-like)', '', '```', 'REGRESSION', sel(q3r).to_string(index=False), 'CLASSIFICATION', sel(q3c).to_string(index=False), '```', '',
          '## Q1/Q4 paired contrasts and overall statements', '', 'See results_v2/summary_overall_8datasets.csv (conformal_overall.py; see journal n0083 for why the frozen script column is not used) and summary_tally_{reg,cls}.csv.', '',
          '## Figures', '', '- results_v2/figures/ratio_reg.png, ratio_cls.png: relative width / set size per method and backbone.', '- results_v2/figures/coverage_novel_vs_familiar.png: coverage on novel versus familiar test molecules.']
    (C / 'CONFORMAL_SUMMARY.md').write_text('\n'.join(md) + '\n')
    print(q5.to_string(index=False)); print(sel(q3r).to_string(index=False))
