#!/bin/bash
# Post-processing after the Uni-Mol extension units: wait for all units, freeze, extended evaluate, comparison, validation predictions, conformal.
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work; V10=$W/expanded_v10_unimol; C=$W/expanded_v5_conformal
PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python; export PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp
cd $V10; F="grep -v Warning|grep -v warn|grep -v Weights"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
# 1. wait for all extension units (or finish them if no worker is active any more)
while true; do
  $PY tierb_unimol_ext.py freeze 2>&1 | grep FROZEN
  n=$(ls frozen_ext 2>/dev/null | wc -l); [ "$n" -ge 8 ] && break
  recent=$(find models/Uni-Mol/scaffold -newermt "-15 minutes" -type f 2>/dev/null | head -1)
  if [ -z "$recent" ]; then
    log "no activity for 15 min, $n of 8 datasets frozen: clearing stale claims and finishing remaining units here"
    for f in claims_ext/*; do b=$(basename $f); r=${b#ext_}; d=${r%%_*}; r=${r#*_}; k=${r%_*}; s=${r##*_}; [ -f models/Uni-Mol/scaffold/$d/$k/seed-$s/summary.json ] || rm -f $f; done
    $PY tierb_unimol_ext.py worker 2>&1 | grep -v "Warning\|warn\|Weights"
  fi
  sleep 120
done
log "all extension units done and frozen"
ls failures_ext 2>/dev/null && log "FAILURES present"
# 2. extended evaluate (opens the test sets for the extension methods)
[ -d results/Uni-Mol_B_backup ] || cp -r results/Uni-Mol results/Uni-Mol_B_backup
$PY tierb_unimol_ext.py evaluate 2>&1 | grep -v "Warning\|warn\|Weights" | tail -20
log "evaluate done"
# 3. sanity: base methods unchanged by the rerun
$PY - <<'E'
import json,glob,numpy as np
bad=0;n=0
for f in glob.glob('results/Uni-Mol_B_backup/scaffold/*/per_seed_metrics.json'):
    d=f.split('/')[-2];a={(r['method'],r['seed']):r for r in json.load(open(f))};b={(r['method'],r['seed']):r for r in json.load(open(f'results/Uni-Mol/scaffold/{d}/per_seed_metrics.json'))}
    for k,r in a.items():
        if k not in b:bad+=1;continue
        for m in ('nll','rmse','roc-auc'):
            if m in r and m in b[k] and r[m] is not None:
                n+=1
                if abs(r[m]-b[k][m])>1e-4*max(1,abs(r[m])):bad+=1;print('DIFF',d,k,m,r[m],b[k][m])
print('base-method rows compared',n,'differences',bad)
E
$PY unimol_compare.py 2>&1 | grep -v "Warning\|warn" | tail -8
log "compare done"
# 4. validation predictions for conformal, then conformal
cd $C
$PY valid_preds_unimol.py 2>&1 | grep -v "Warning\|warn\|Weights" | tail -10
$PY valid_preds_laplace_unimol.py 2>&1 | grep -v "Warning\|warn\|Weights" | tail -10
log "validation predictions done"
$PY conformal_unimol.py selftest 2>&1 | tail -3
$PY conformal_unimol.py run 2>&1 | tail -3
log "POST-PROCESSING DONE"
