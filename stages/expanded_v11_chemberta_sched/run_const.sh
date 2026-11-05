#!/bin/bash
# authors' unmodified ChemBERTa trainer, paper settings, DEFAULT constant scheduler (no --lr_scheduler_type), usage: run_const.sh DATASET SEEDS...
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work; PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python; D=$1; shift
OUT=$W/expanded_v11_chemberta_sched/authors_const_out
for S in "$@"; do
  [ -d $OUT/$D/ChemBERTa/none/seed-$S/preds ] && continue
  ( cd $W/repos/MUBen && MPLCONFIGDIR=/tmp PYTHONPATH=. TOKENIZERS_PARALLELISM=false $PY ./run/run.py --descriptor_type Linear --data_folder ./data/files --pretrained_model_name_or_path $W/expanded_v3/chemberta_checkpoint --lr 0.00001 --batch_size 128 --n_epochs 200 --valid_tolerance 40 --regression_with_variance --dataset_name $D --uncertainty_method none --seed $S --result_folder $OUT --disable_wandb > $OUT.$D.$S.log 2>&1 )
done
