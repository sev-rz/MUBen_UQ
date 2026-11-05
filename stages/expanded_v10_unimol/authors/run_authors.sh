#!/bin/bash
# usage: run_authors.sh DATASET SEED EPOCHS OUTROOT   (authors' unmodified run/unimol.py, paper settings)
D=$1; S=$2; EP=${3:-100}; ROOT=${4:-/l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v10_unimol/authors_out}
W=/l/users/sevinch.zoda/schub/projects/muben-laplace/work; PY=/l/users/sevinch.zoda/schub/repo-envs/4b5817be54ed5792/bin/python
case $D in esol|freesolv|lipo|qm7) RV=true;; *) RV=false;; esac
CFG=$ROOT/$D/seed-$S/config.yaml; mkdir -p $ROOT/$D/seed-$S
[ -f $ROOT/$D/seed-$S/DONE ] && exit 0
cat > $CFG <<Y
dataset_name: "$D"
data_folder: "$W/expanded_v4/data"
unimol_feature_folder: "$W/data/unimol_archive/extracted/molecular_property_prediction"
checkpoint_path: "$W/data/unimol_ckpt/mol_pre_no_h_220816.pt"
result_folder: "$ROOT/$D/seed-$S/out"
ignore_preprocessed_dataset: true
num_preprocess_workers: 1
retrain_model: true
lr: 0.00005
batch_size: 128
n_epochs: $EP
valid_tolerance: 40
lr_scheduler_type: "linear"
warmup_ratio: 0.1
regression_with_variance: $RV
seed: $S
uncertainty_method: "none"
disable_wandb: true
disable_result_saving: false
Y
cd $W/repos/MUBen && PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY run/unimol.py $CFG > $ROOT/$D/seed-$S/log.txt 2>&1 && touch $ROOT/$D/seed-$S/DONE
