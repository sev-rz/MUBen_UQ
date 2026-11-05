"""Descriptive digest behind PHASE3_SUMMARY.md, computed only from the result CSVs/JSONs.

Agreement counts (|delta| <= 1 or 2 published SDs), the random-vs-scaffold gap (unpaired:
different test molecules), and paired within-split extension contrasts on random splits.
"""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent;R=ROOT/'results';V2=ROOT.parent/'expanded_v2'/'results'
def rows(p):return list(csv.DictReader(open(p)))
def agreement(name,items,delta,sd):
    v=[(float(r[delta]),float(r[sd])) for r in items if r[sd] not in ('',None)]
    print(f"{name}: {len(items)} rows, {len(v)} with published SD; within 1 SD {sum(abs(d)<=s for d,s in v)}, within 2 SD {sum(abs(d)<=2*s for d,s in v)}")
    return [r for r in items if r[sd] not in ('',None) and abs(float(r[delta]))>2*float(r[sd])]
out={}
far=agreement('random vs published random',rows(R/'random_split_comparison.csv'),'delta_reproduced_minus_upstream','upstream_std')
for r in far:print('  >2SD',r['dataset'],r['method'],r['metric'],f"paper {float(r['upstream_mean']):.3f}±{float(r['upstream_std']):.3f} ours {float(r['reproduced_mean']):.3f}")
agreement('v2 scaffold vs published scaffold',rows(V2/'original_comparison.csv'),'difference','original_std')
b=[r for r in rows(R/'bayesian_comparison.csv') if r['metric'] in ('rmse','mae','nll','ce_upstream')]
agreement('BBP/SGLD (definition-matched metrics)',b,'delta_reproduced_minus_upstream','upstream_std')
# Random vs scaffold (unpaired): v3 random-split means vs v2 scaffold means, same code and members.
rs={(r['dataset'],r['method']):r for r in rows(R/'random_split_summary.csv')};ss={(r['dataset'],r['method']):r for r in rows(V2/'summary_metrics.csv')}
for d,m in [('bace','roc_auc'),('bbbp','roc_auc'),('esol','rmse')]:
    print(f"split gap {d} Deterministic {m}: random {float(rs[(d,'Deterministic')][m+'_mean']):.3f} scaffold {float(ss[(d,'Deterministic')][m+'_mean']):.3f}")
# Paired within-split extension contrasts on random splits (same test molecules within a split).
per=rows(R/'random_split_per_split_metrics.csv')
for d in ('bace','bbbp','esol'):
    for m,ref in [('Novelty Laplace','Laplace'),('Laplace','Deterministic'),('Novelty Laplace','Deterministic')]:
        diffs=[float(next(r['nll'] for r in per if r['dataset']==d and r['method']==m and r['split']==str(s)))-float(next(r['nll'] for r in per if r['dataset']==d and r['method']==ref and r['split']==str(s))) for s in (0,1,2)]
        print(f"paired NLL {d} {m} - {ref}: " + ' '.join(f'{x:+.3f}' for x in diffs))
al=rows(R/'active_learning/per_seed_endpoints.csv')
print('AL random expectation per seed:',' '.join(f"{float(r['random_expectation_150']):.1f}" for r in al if r['strategy']=='random'),'| initial positives:',' '.join(r['initial_positives'] for r in al if r['strategy']=='random'))
for r in rows(R/'active_learning/summary.csv'):
    print(f"AL {r['strategy']:9s} pos {float(r['positives_150_mean']):.1f} [{float(r['positives_150_min']):.0f},{float(r['positives_150_max']):.0f}] distinct {float(r['distinct_active_scaffolds_mean']):.1f} new {float(r['new_active_scaffolds_mean']):.1f} acyclic {float(r['acyclic_actives_mean']):.1f} audc {float(r['audc_mean']):.1f} testAUC {float(r['test_auc_mean']):.3f} NLL {float(r['test_nll_mean']):.3f} wall {float(r['wall_seconds_mean']):.0f}s")
