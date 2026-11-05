"""GPU-correct launcher for valid_preds.py ChemBERTa (valid_preds.py itself is unchanged).
Why: valid_preds.py imports tierb_dnn first, and tierb_dnn sets CUDA_VISIBLE_DEVICES='' (the DNN is CPU by design); tierb_chemberta, imported later,
then 'restores' the already hidden value, so ChemBERTa validation predictions silently ran on CPU (5-20 min per group instead of 10-90 s on the GPU).
Here tierb_chemberta is imported first (it captures the real CUDA_VISIBLE_DEVICES before tierb_dnn hides it and restores it afterwards), then
valid_preds.py runs as __main__ with the same arguments, and the device is asserted.
  python valid_preds_gpu.py ChemBERTa [splits...] [--datasets ...]"""
import sys, os, runpy
from pathlib import Path
C = Path(__file__).resolve().parent; V = C.parent / 'expanded_v4'
sys.path.insert(0, str(V)); os.chdir(V)
import tierb_chemberta as cb        # must come before tierb_dnn
import torch
assert torch.cuda.is_available() and cb.DEV.type == 'cuda', 'no GPU: refusing to run ChemBERTa validation predictions on CPU'
print('device', torch.cuda.get_device_name(), flush=True)
sys.argv = [str(C / 'valid_preds.py')] + sys.argv[1:]
runpy.run_path(str(C / 'valid_preds.py'), run_name='__main__')
