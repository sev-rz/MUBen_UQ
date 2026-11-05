#!/bin/bash
# Authors' unmodified run/run.py (MUBen f0ec556) with their scripts/config-example.yaml settings; DNN-rdkit, Deterministic, regression with variance.
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work/authors_code_check
PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python
cd /l/users/sevinch.zoda/schub/projects/muben-laplace/work/repos/MUBen
OUT=/l/users/sevinch.zoda/schub/projects/muben-laplace/work/authors_code_check/out
for ds in esol qm7; do for seed in 0 1 2; do
  if [ -f $OUT/$ds/DNN-rdkit/none/seed-$seed/DONE ]; then continue; fi
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="." $PY ./run/run.py --descriptor_type RDKit --data_folder ./data/files --dataset_name $ds \
    --uncertainty_method none --regression_with_variance --lr 0.0001 --batch_size 128 --n_epochs 200 --valid_tolerance 50 --seed $seed \
    --result_folder $OUT --disable_wandb > $W/log_${ds}_${seed}.txt 2>&1 && touch $OUT/$ds/DNN-rdkit/none/seed-$seed/DONE
done; done
echo ALL_DONE > $W/ALL_DONE
