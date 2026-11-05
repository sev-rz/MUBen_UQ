"""Check C (n0095): recompute the 200 normalised RDKit descriptors in the authors' environment (RDKit 2023.09.5, descriptastorus 2.6.1)
and compare with the cached Tier-B descriptors (RDKit 2024.09.6, descriptastorus 2.8.0). Same procedure as tierb_dnn.table(): process()[1:], NaN -> 0, unparsable -> zeros."""
import csv, json, sys, numpy as np
from pathlib import Path
from descriptastorus.descriptors import rdNormalizedDescriptors
import rdkit
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V4=W/'expanded_v4'
gen=rdNormalizedDescriptors.RDKit2DNormalized();out={}
print('rdkit',rdkit.__version__,flush=True)
for d in ['bace','bbbp','clintox','esol','freesolv','lipo','qm7','tox21']:
    sm=[r['smiles'] for p in ['train','valid','test'] for r in csv.DictReader((V4/'data'/d/f'{p}.csv').open())]
    old=np.load(V4/'features'/f'{d}.npz',allow_pickle=False)['x'].astype(np.float64);new=np.zeros_like(old);unparsed_new=[]
    for i,s in enumerate(sm):
        v=gen.process(s)
        if v is None:new[i]=0;unparsed_new.append(i);continue
        x=np.asarray(v[1:],np.float64);x[np.isnan(x)]=0;new[i]=x
    diff=np.abs(old-new);tol=1e-3*np.maximum(np.maximum(np.abs(old),np.abs(new)),0.01)
    bad=diff>tol;rows=bad.any(1);zero_old=(np.abs(old).sum(1)==0)
    out[d]={'molecules':len(sm),'frac_molecules_any_feature_differs':float(rows.mean()),'n_molecules_differ':int(rows.sum()),'features_affected':int(bad.any(0).sum()),
            'max_abs_diff':float(diff.max()),'p99_abs_diff':float(np.percentile(diff,99)),'rows_zero_in_tierb':int(zero_old.sum()),'rows_unparsed_here':len(unparsed_new),
            'material_gt5pct':bool(rows.mean()>0.05)}
    print(d,json.dumps(out[d]),flush=True)
json.dump(out,open(W/'expanded_v9_checks'/'descriptor_drift.json','w'),indent=1)
