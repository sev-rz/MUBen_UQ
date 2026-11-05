# Execution record

All scientific computation used Slurm217442, one CPU,4GB, no GPU, through SC Hub on ws-l5-032. Persistent project root is /l/users/sevinch.zoda/schub/projects/muben-laplace.

- c0012: initial fitting; stopped on missing regex dependency after saving six models.
- c0013: installed regex2024.11.6; completed fitting and FROZEN.json.
- c0014: first evaluation stopped on SWAG covariance buffer reload shape.
- c0015: shape-aware loader strictly restored tensors; all93 evaluations completed; later source-scaler check needed explicit numeric casting.
- c0016: analysis check required a numeric cast for the scaler output as well as its input.
- c0017: all93 prediction rows recomputed,88 original comparisons generated, source checks and plots passed.

These operational repairs did not retrain models, change frozen hyperparameters, or select settings from test metrics. Original failed logs are retained. The final packaging cell is recorded in the parent SC Hub journal.

Report:6pages, rendered and visually checked locally. Plotting and numerical analysis run on allocated compute, not the login node. Portable archive has its own checksum; its manifest covers raw data, code, models, predictions and report.
