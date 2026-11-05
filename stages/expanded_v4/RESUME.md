
## ChemBERTa (launched 2026-10-08T17:12:53Z by chat, GPU env work/.venv-gpu)
- Measured (bench_gpu, job 223265): Tox21 2.25 s/epoch on 1 RTX 5000 Ada (CPU was 148 s); 3-6 processes/GPU add only ~10% throughput -> 2 per GPU.
- Workers: gpu partition job 223273, ws-ia job 223274 (23.5 h wall). Both run slurm/cb_worker.sh and share the unit queue via claims/.
- Finish (freeze + evaluate, after both): job 223275 (slurm/cb_finish.sh).
- If ws-ia job 223274 times out: resubmit the same sbatch line; completed units are skipped. If a worker crashed mid-unit, delete its
  claim file claims/ChemBERTa_<split>_<dataset>_<kind>_<seed> (only if that unit has no summary.json) and resubmit.
- Relaunch lines:
  sbatch -p gpu --gres=gpu:1 -c 6 --mem=40G -J cb-tierb-gpu -o slurm/cb_gpu_%j.out slurm/cb_worker.sh
  sbatch -p ws-ia --gres=gpu:1 -c 6 --mem=40G -t 23:30:00 -J cb-tierb-wsia -o slurm/cb_wsia_%j.out slurm/cb_worker.sh
  sbatch -p gpu --gres=gpu:1 -c 4 --mem=32G -J cb-tierb-finish -o slurm/cb_finish_%j.out slurm/cb_finish.sh
