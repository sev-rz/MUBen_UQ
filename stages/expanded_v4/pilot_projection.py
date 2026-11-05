"""ChemBERTa GPU projection from CHEMBERTA_PILOT_UNITS.json; adds it to PILOT_TIMING.json (DNN part already there).

Cost of a fit ~ train rows x epochs x seconds-per-row-epoch (pilot, incl. per-epoch validation) + model load.
Typical = the pilot's mean epochs-to-stop for every fit; worst = 200 epochs for every fit. SGLD adds 60 epochs, SWAG 20.
"""
import json
from pathlib import Path
import numpy as np
import tierb_dnn as t
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'CHEMBERTA_PILOT_UNITS.json').read_text())['units']
rate=float(np.mean([u['mean_epoch_seconds']/u['train_n'] for u in p.values()]))
typical_epochs=float(np.mean([u['epochs'] for u in p.values()]));load_s=10.
n_train={d:int((t.table(d)['part']==0).sum()) for d in t.DATASETS}
extra=lambda k:60 if k=='sgld' else 20 if k=='posthoc' else 0
def hours(epochs,splits):
    return sum(n_train[d]*rate*(epochs+extra(k))+load_s for sp in splits for d in t.DATASETS for k,_ in t.units(sp,d))/3600
proj={'pilot':p,'seconds_per_train_row_epoch':rate,'typical_epochs':typical_epochs,
      'gpu_hours_typical':{'scaffold':hours(typical_epochs,['scaffold']),'random':hours(typical_epochs,t.SPLITS[1:])},
      'gpu_hours_worst_case':{'scaffold':hours(200,['scaffold']),'random':hours(200,t.SPLITS[1:])},
      'units':{'scaffold':sum(len(t.units('scaffold',d)) for d in t.DATASETS),'random':sum(len(t.units(s,d)) for s in t.SPLITS[1:] for d in t.DATASETS)},
      'gate':'launch if projected total <= 4 GPU-days (96 GPU-hours), else stop and report options',
      'note':'sequential GPU time; running 3 processes on the GPU can only shorten wall time'}
proj['gpu_days_typical_total']=sum(proj['gpu_hours_typical'].values())/24;proj['gpu_days_worst_total']=sum(proj['gpu_hours_worst_case'].values())/24
timing=json.loads((ROOT/'PILOT_TIMING.json').read_text());timing['ChemBERTa']=proj
(ROOT/'PILOT_TIMING.json').write_text(json.dumps(timing,indent=1))
print(json.dumps({k:v for k,v in proj.items() if k!='pilot'},indent=0,default=lambda x:round(x,3)))
