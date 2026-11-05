"""Metrics of the authors' GIN runs (their own functions) for the registered bridge check; reads only the authors' prediction files."""
import sys,glob,numpy as np,pandas as pd
sys.path.insert(0,'/l/users/sevinch.zoda/schub/projects/muben-laplace/work/repos/MUBen')
from muben.utils.io import load_results
from muben.utils.metrics import regression_metrics
rows=[]
for ds in ['esol','qm7']:
    for seed in [0,1,2]:
        p=glob.glob(f'bridge_authors_out/{ds}/GIN/none/seed-{seed}/preds/*.pt')
        if not p:continue
        preds,var,lbs,masks=load_results(p);m=regression_metrics(preds,var,lbs,masks)
        rows.append({'dataset':ds,'seed':seed,**{k:v['macro-avg'] for k,v in m.items()}})
d=pd.DataFrame(rows);d.to_csv('bridge_authors_gin_metrics.csv',index=False);print(d.round(4).to_string(index=False))
