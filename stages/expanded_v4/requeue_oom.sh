#!/bin/bash
# Move CUDA-OOM failure records of ChemBERTa units without summary.json to failures_archive/ and release their claims (unit is re-run from its seed).
cd /l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v4; ts=$(date -u +%Y%m%dT%H%MZ); n=0
for f in failures/ChemBERTa_*.txt; do
  [ -f "$f" ] || continue; grep -q OutOfMemoryError "$f" || { echo "non-OOM failure kept: $f"; continue; }
  u=$(basename "$f" .txt); u=${u#ChemBERTa_}; split=${u%%_*}; r=${u#*_}; d=${r%%_*}; r=${r#*_}; kind=${r%%_*}; seed=${r#*_}
  [ -f models/ChemBERTa/$split/$d/$kind/seed-$seed/summary.json ] && { echo "has summary, leaving: $u"; continue; }
  mkdir -p failures_archive/oom_$ts; mv "$f" failures_archive/oom_$ts/; command rm -f claims/ChemBERTa_$u; n=$((n+1))
done; echo "requeued $n units (archive failures_archive/oom_$ts)"
