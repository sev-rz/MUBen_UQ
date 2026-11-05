"""Descriptive NLL/CE diagnostic on SAVED Tier-B test predictions. No training, no tuning, frozen files untouched.
Independent numpy recomputation of Gaussian NLL, convention variants, and variance calibration (z-scores, coverage)."""
import glob, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
R=Path('expanded_v4/results'); OUT=Path('expanded_v4/diagnostics_nll'); REG=['esol','freesolv','lipo','qm7']
rows=[]
for npz in sorted(R.glob('*/*/*/*.npz')):
    bb,split,ds=npz.parts[-4:-1]
    if ds not in REG: continue
    method,seed=npz.stem.rsplit('-',1)
    z=np.load(npz,allow_pickle=True); k=z['masks'][:,0].astype(bool)
    y=z['labels'][k,0].astype(float); mu=z['mean'][k,0].astype(float); v=z['variance'][k,0].astype(float)
    err=y-mu; n=len(y)
    nll_src=np.mean(0.5*(np.log(v)+err**2/v))              # F.gaussian_nll_loss default: no constant, mean
    sd=np.sqrt(v); zs=err/sd
    rows.append(dict(backbone=bb,split=split,dataset=ds,method=method.replace('_',' '),seed=seed,n=n,
        rmse=np.sqrt(np.mean(err**2)),mean_var=v.mean(),mse=np.mean(err**2),
        nll_recomputed=nll_src,nll_full=nll_src+0.5*np.log(2*np.pi),
        nll_sd_scaled=nll_src-np.log(np.std(y)),             # NLL if labels were standardised by test-label SD (convention probe)
        z_mean=zs.mean(),z_sd=zs.std(),cov90=np.mean(np.abs(zs)<=norm.ppf(0.95)),cov50=np.mean(np.abs(zs)<=norm.ppf(0.75)),
        logvar_mean=np.log(v).mean(),label_sd=np.std(y)))
d=pd.DataFrame(rows); d.to_csv(OUT/'per_group_recomputed.csv',index=False)
# 1) does the independent recomputation equal the stored metric?
chk=[]
for p in sorted(R.glob('*/*/*/per_seed_metrics.json')):
    bb,split,ds=p.parts[-4:-1]
    if ds not in REG: continue
    for r in json.loads(p.read_text()):
        chk.append(dict(backbone=bb,split=split,dataset=ds,method=r['method'],seed=str(r['seed']),nll_stored=r['nll'],ce_stored=r['ce']))
c=pd.DataFrame(chk); d['seed']=d['seed'].astype(str); m=d.merge(c,on=['backbone','split','dataset','method','seed'])
m['abs_diff']=(m.nll_recomputed-m.nll_stored).abs(); m.to_csv(OUT/'recompute_vs_stored.csv',index=False)
print('groups',len(d),'matched to stored',len(m),'| max|NLL recomputed - stored| =',float(m.abs_diff.max()))
print('worst 3:');print(m.sort_values('abs_diff',ascending=False).head(3)[['backbone','split','dataset','method','seed','nll_recomputed','nll_stored']].to_string(index=False))
