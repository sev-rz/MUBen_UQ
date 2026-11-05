#!/bin/bash
D=/l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v13_schedule; PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python; export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $D
for s in 0 100 200; do ( $PY run_lin_seed.py tox21 $s 2>&1 | grep --line-buffered '^{' ) & done
wait; $PY merge_lin.py tox21; echo "B fits done"
