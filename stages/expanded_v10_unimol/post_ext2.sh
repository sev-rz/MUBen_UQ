#!/bin/bash
# second pass: the first extended evaluate skipped all datasets (per_seed_metrics.json already existed). Move the base results aside and recompute with all methods.
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work; V10=$W/expanded_v10_unimol; C=$W/expanded_v5_conformal
PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python; export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp
cd $V10; log(){ echo "[$(date +%H:%M:%S)] $*"; }
[ -d results/Uni-Mol_pre_ext ] || mv results/Uni-Mol results/Uni-Mol_pre_ext
$PY tierb_unimol_ext.py evaluate 2>&1 | grep -v "Warning\|warn\|Weights" | tail -12
log "evaluate done"
$PY - <<'E'
import json,glob
bad=0;n=0
for f in glob.glob('results/Uni-Mol_pre_ext/scaffold/*/per_seed_metrics.json'):
    d=f.split('/')[-2];a={(r['method'],r['seed']):r for r in json.load(open(f))};b={(r['method'],r['seed']):r for r in json.load(open(f'results/Uni-Mol/scaffold/{d}/per_seed_metrics.json'))}
    for k,r in a.items():
        if k not in b:bad+=1;print('MISSING',d,k);continue
        for m in ('nll','rmse','roc-auc'):
            if m in r and m in b[k] and r[m] is not None:
                n+=1
                if abs(r[m]-b[k][m])>1e-4*max(1,abs(r[m])):bad+=1;print('DIFF',d,k,m,r[m],b[k][m])
    print(d,'methods now',sorted(set(k[0] for k in b)))
print('base-method rows compared',n,'differences',bad)
E
$PY unimol_compare.py 2>&1 | grep -v "Warning\|warn" | tail -8
log "compare done"
cd $C; rm -rf results_unimol selftest_unimol
$PY conformal_unimol.py selftest 2>&1 | tail -2
$PY conformal_unimol.py run 2>&1 | tail -2
log "POST2 DONE"
