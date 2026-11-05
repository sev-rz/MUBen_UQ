#!/bin/bash
D=/l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v13_schedule; PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python; export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $D; log(){ echo "[$(date +%H:%M:%S)] $*"; }
( $PY run_lin.py esol bbbp 2>&1 | grep --line-buffered '^{' ) &
( $PY run_lin.py clintox 2>&1 | grep --line-buffered '^{' ) &
wait; log "A fits done"
while [ $(ls metrics/lin_det_*.json 2>/dev/null | wc -l) -lt 8 ]; do sleep 20; done
log "all 8 datasets fitted; Laplace stage"
cd laplace && $PY chemberta_laplace_lin.py all 2>&1 | grep -v "Warning\|warn" | tail -30
log "LAPLACE DONE"
cd $D && $PY det_preds_lin.py 2>&1 | grep -v "Warning\|warn" | tail -9
log "det preds saved"
$PY analyze_sched.py 2>&1 | grep -v "Warning\|warn"
log "ANALYSIS DONE"
