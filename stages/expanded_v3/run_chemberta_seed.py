"""Run one ChemBERTa seed of the unchanged train_chemberta.fit (parallel per-seed jobs)."""
import sys, train_chemberta as t
t.SEEDS=[int(sys.argv[1])]; t.fit()
