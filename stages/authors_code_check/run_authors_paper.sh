#!/bin/bash
# Authors' unmodified run/run.py with the DNN settings stated in the paper (Appendix B): 8 hidden 128-wide layers (+ input layer = 9 linear), GELU,
# dropout 0.1, batch 256, constant lr 2e-4, 400 epochs, early-stopping tolerance 50, AdamW wd 0.01 (arg default checked). DNN-rdkit, Deterministic.
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work/authors_code_check
PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python
cd /l/users/sevinch.zoda/schub/projects/muben-laplace/work/repos/MUBen
OUT=$W/out_paper
for ds in esol qm7; do for seed in 0 1 2; do
  [ -f $OUT/$ds/DNN-rdkit/none/seed-$seed/DONE ] && continue
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="." $PY ./run/run.py --descriptor_type RDKit --data_folder ./data/files --dataset_name $ds \
    --uncertainty_method none --regression_with_variance --activation GELU --lr 0.0002 --batch_size 256 --n_epochs 400 --valid_tolerance 50 --seed $seed \
    --result_folder $OUT --disable_wandb > $W/paperlog_${ds}_${seed}.txt 2>&1 && touch $OUT/$ds/DNN-rdkit/none/seed-$seed/DONE
done; done
echo ALL_DONE > $W/ALL_DONE_paper
