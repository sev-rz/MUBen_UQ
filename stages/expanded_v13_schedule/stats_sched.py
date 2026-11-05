"""Item 9 statistics (check-E design): per-molecule paired NLL Laplace minus Deterministic under the linear schedule, bootstrap over test molecules (2000), exact sign-flip over the eight dataset z-values. Exploratory."""
import json,itertools,numpy as np
from pathlib import Path
D=Path(__file__).resolve().parent;DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'};rng=np.random.default_rng(20261012);B=2000
def nll(z,reg):
    y=z['labels'].astype(float);m=z['masks'].astype(bool);mu=z['mean'].astype(float)
    if reg:v=z['variance'].astype(float);return 0.5*(np.log(v)+(y-mu)**2/v),m
    with np.errstate(divide='ignore'):lp=np.maximum(np.log(mu),-100.);l1=np.maximum(np.log(1-mu),-100.)
    return -(y*lp+(1-y)*l1),m
def ok_tasks(z,reg):
    if reg:return np.ones(z['labels'].shape[1],bool)
    y=z['labels'].astype(float);m=z['masks'].astype(bool);return np.array([0<y[m[:,j],j].mean()<1 for j in range(y.shape[1])])
def macro(x,m,ok,idx=None):
    x0=np.where(m,x,0.);
    if idx is None:return float(np.mean([x0[:,j].sum()/m[:,j].sum() for j in range(x.shape[1]) if ok[j]]))
    S=x0[idx].sum(1);N=m[idx].sum(1);return np.mean([S[:,j]/N[:,j] for j in range(x.shape[1]) if ok[j]],axis=0)
rows=[];zs=[]
for d in DS:
    reg=d in REG;p=D/'laplace'/'results'/'scaffold'/d;diffs=[]
    for s in [0,100,200]:
        zd=np.load(p/f'Deterministic-{s}.npz');zl=np.load(p/f'Laplace-{s}.npz');nd,m=nll(zd,reg);nl,_=nll(zl,reg);diffs.append((nl-nd,m,ok_tasks(zd,reg)))
    n=diffs[0][0].shape[0];idx=rng.integers(0,n,(B,n));boot=np.mean([macro(x,m,ok,idx) for x,m,ok in diffs],axis=0);est=np.mean([macro(x,m,ok) for x,m,ok in diffs])
    lo,hi=np.percentile(boot,[2.5,97.5]);se=float(boot.std());ident=abs(est)<1e-4 and se<1e-4;z=0.0 if ident else est/se
    rows.append((d,round(est,4),round(lo,4),round(hi,4),'identical' if ident else ('improves' if hi<0 else ('worsens' if lo>0 else 'unclear'))));zs.append(z)
for r in rows:print(r)
z=np.array(zs);obs=z.sum();flips=np.array([np.sum(z*np.array(s)) for s in itertools.product([1,-1],repeat=8)]);print('sum z',round(float(obs),2),'exact one-sided sign-flip p',round(float((flips<=obs+1e-12).mean()),4))
json.dump({'rows':rows,'p':float((flips<=obs+1e-12).mean())},open(D/'stats_sched.json','w'))
