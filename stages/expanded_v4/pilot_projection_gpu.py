"""ChemBERTa GPU projection from the production units finished so far (the CPU-run pilot 222938 was invalid, journal n0029);
writes PILOT_TIMING.json['ChemBERTa']. Cost of a fit ~ train rows x epochs x seconds-per-row-epoch (measured with 2 processes per GPU,
validation pass included) + model load. Typical = mean epochs-to-stop of the finished fits; worst = 200 epochs for every fit.
SGLD adds 60 epochs, SWAG 20. GPU-hours = process-hours / 2 (two processes share one GPU)."""
import json, glob
from pathlib import Path
import numpy as np
import tierb_dnn as t
ROOT=Path(__file__).resolve().parent
n_train={d:int((t.table(d)['part']==0).sum()) for d in t.DATASETS}
rows=[]
for p in glob.glob(str(ROOT/'models/ChemBERTa/*/*/*/seed-*/summary.json')):
    j=json.loads(Path(p).read_text())
    if 'mean_epoch_seconds' in j:rows.append((j['dataset'],j['kind'],j['epochs'],j['mean_epoch_seconds']))
by={d:[r for r in rows if r[0]==d] for d in t.DATASETS}
rate_d={d:float(np.mean([r[3]/n_train[d] for r in v])) for d,v in by.items() if v}
rate=float(np.mean(list(rate_d.values())));typical_epochs=float(np.mean([r[2] for r in rows]));load_s=10.
extra=lambda k:60 if k=='sgld' else 20 if k=='posthoc' else 0
def hours(epochs,splits):
    return sum(n_train[d]*rate_d.get(d,rate)*(epochs+extra(k))+load_s for sp in splits for d in t.DATASETS for k,_ in t.units(sp,d))/3600/2
proj={'source':'production units finished so far (GPU, .venv-gpu, 2 processes per GPU); replaces the invalid CPU pilot',
      'finished_fits_used':len(rows),'datasets_measured':sorted(rate_d),'seconds_per_train_row_epoch_by_dataset':rate_d,'seconds_per_train_row_epoch_mean':rate,
      'typical_epochs':typical_epochs,'n_train':n_train,
      'benchmark':{'job':223265,'dataset':'tox21','seconds_per_epoch_1_process':2.25,'seconds_per_epoch_6_processes_per_gpu':10.65,'note':'3-6 processes per GPU add only ~10-30% throughput'},
      'gpu_hours_typical':{'scaffold':hours(typical_epochs,['scaffold']),'random':hours(typical_epochs,t.SPLITS[1:])},
      'gpu_hours_worst_case':{'scaffold':hours(200,['scaffold']),'random':hours(200,t.SPLITS[1:])},
      'units':{'scaffold':sum(len(t.units('scaffold',d)) for d in t.DATASETS),'random':sum(len(t.units(s,d)) for s in t.SPLITS[1:] for d in t.DATASETS)},
      'gate':'launch if projected total <= 4 GPU-days (96 GPU-hours), else stop and report options'}
proj['gpu_days_typical_total']=sum(proj['gpu_hours_typical'].values())/24;proj['gpu_days_worst_total']=sum(proj['gpu_hours_worst_case'].values())/24
proj['gate_passed']=bool(proj['gpu_days_typical_total']<=4)
proj['wall_hours_typical_on_2_gpus']=sum(proj['gpu_hours_typical'].values())/2;proj['wall_hours_worst_on_2_gpus']=sum(proj['gpu_hours_worst_case'].values())/2
timing=json.loads((ROOT/'PILOT_TIMING.json').read_text());timing['ChemBERTa']=proj
(ROOT/'PILOT_TIMING.json').write_text(json.dumps(timing,indent=1))
print(json.dumps({k:v for k,v in proj.items() if k not in('n_train','benchmark','source')},indent=0,default=lambda x:round(x,3)))
