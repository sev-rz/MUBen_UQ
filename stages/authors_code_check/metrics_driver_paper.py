"""Metrics from the authors' own functions (muben.utils.io.load_results, muben.utils.metrics.regression_metrics) on the predictions
saved by the authors' run/run.py. Used because assist/results_get_metrics.py looks for 'preds-preds' while the trainer writes 'preds'."""
import sys, glob, numpy as np, pandas as pd
from muben.utils.io import load_results
from muben.utils.metrics import regression_metrics
W='/l/users/sevinch.zoda/schub/projects/muben-laplace/work/authors_code_check'
rows=[]
for ds in sys.argv[1:]:
    for seed in [0,1,2]:
        p=glob.glob(f'{W}/out_paper/{ds}/DNN-rdkit/none/seed-{seed}/preds/*.pt')
        if not p: continue
        preds,var,lbs,masks=load_results(p)
        m=regression_metrics(preds,var,lbs,masks)
        r={'dataset':ds,'seed':seed,**{k:v['macro-avg'] for k,v in m.items()}}
        # our independent check on the same arrays: mean predicted variance vs MSE
        k=masks.reshape(-1).astype(bool); y=np.asarray(lbs).reshape(-1)[k]; mu=np.asarray(preds).reshape(-1)[k]; v=np.asarray(var).reshape(-1)[k]
        r['mean_var']=float(v.mean()); r['mse']=float(((y-mu)**2).mean()); r['z_sd']=float(((y-mu)/np.sqrt(v)).std())
        rows.append(r)
d=pd.DataFrame(rows); d.to_csv(f'{W}/authors_paper_metrics_'+'_'.join(sys.argv[1:])+'.csv',index=False)
pd.set_option('display.width',200); print(d.round(4).to_string(index=False))
print(d.groupby('dataset').agg(['mean','std']).round(4).T.to_string())
