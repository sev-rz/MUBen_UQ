"""SENSITIVITY: Laplace on Tier B with a wider alpha grid 1e-8..1e4 (registered in journal n0061). Same code path as tierb_laplace.py with ALPHAS replaced;
outputs under sensitivity_wide_grid/. Stages: fit (train+valid only, writes markers) then evaluate (test, plain Laplace only, behind the markers)."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tierb_laplace as L
import numpy as np, torch
t,e=L.t,L.e
S=L.V6/'sensitivity_wide_grid';GRID=[10.**a for a in range(-8,5)];L.ALPHAS=GRID
FALL=[0];_orig=L.cov_from
def safe_cov(H,alpha):
    A=H+alpha*torch.eye(len(H),dtype=H.dtype)
    try:return torch.cholesky_inverse(torch.linalg.cholesky(A))
    except Exception:
        FALL[0]+=1;w,V=torch.linalg.eigh(A);return (V/w.clamp_min(alpha))@V.T
L.cov_from=safe_cov
GSHA=hashlib.sha256(json.dumps(GRID).encode()).hexdigest()[:16]

def fit():
    for split in t.SPLITS:
        for d in t.DATASETS:
            mk=S/'frozen'/f'DNN_{split}_{d}.json'
            if mk.exists():continue
            c=None;edge=[];n=0
            for s in t.seeds(split)[0]:
                f=S/'results'/split/d/f'seed-{s}'
                if not (f/'fit.json').exists():
                    c=c or t.Ctx(split,d);arr,rep=L.fit_group(c,s);f.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(f/'laplace_posterior.npz',**arr);e.save_json(f/'fit.json',rep)
                rep=json.loads((f/'fit.json').read_text())
                for r in rep['per_task']:
                    n+=1
                    if r['alpha'] in (GRID[0],GRID[-1]):edge.append({'seed':s,'task':r['task'],'alpha':r['alpha']})
            mk.parent.mkdir(parents=True,exist_ok=True)
            e.save_json(mk,{'at_utc':e.now(),'split':split,'dataset':d,'grid':GRID,'grid_sha16':GSHA,'protocol_sha256':e.digest(L.V6/'PROTOCOL.md'),'registration':'n0061',
                            'task_fits':n,'alpha_at_grid_edge':edge,'cholesky_fallbacks_so_far':FALL[0],'test_unopened_by_laplace':True})
    print('FIT DONE; markers',len(list((S/'frozen').glob('*.json'))),'| cholesky fallbacks this process',FALL[0])

def evaluate():
    for split in t.SPLITS:
        for d in t.DATASETS:
            mk=S/'frozen'/f'DNN_{split}_{d}.json';res=S/'results'/split/d
            if not mk.exists() or (res/'per_seed_metrics.json').exists():continue
            c=t.Ctx(split,d);rows=[]
            for s in t.seeds(split)[0]:
                net,_=t.fit(c,'det',s)
                with np.load(res/f'seed-{s}'/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
                mu,v,noise=L.part_outputs(c,net,arr,'test');pm,pv=L.compose(c,mu,v,noise,None,None,0.)
                mean,var=L._convert(c,pm,pv);assert np.isfinite(mean).all() and (var is None or (np.isfinite(var).all() and (var>0).all()))
                rows.append({'backbone':'DNN','split':split,'dataset':d,'method':'Laplace wide','seed':s,'subgroup':'all','n':len(mean),**L.ev.macro(c,mean,var)})
            e.save_json(res/'per_seed_metrics.json',rows)
    print('EVALUATE DONE')

if __name__=='__main__':
    {'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
