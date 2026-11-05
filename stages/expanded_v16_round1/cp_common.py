import csv, math, hashlib, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger; from rdkit.Chem.Scaffolds import MurckoScaffold; RDLogger.DisableLog('rdApp.*')
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V4=W/'expanded_v4';V5=W/'expanded_v5_conformal';V8=W/'expanded_v8_gin'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'};ALPHA=0.1;MINCAL=20
def qhat(s,a=ALPHA):
    s=np.sort(s);n=len(s);k=math.ceil((n+1)*(1-a));return np.inf if k>n else s[k-1]
def wqhat(s,w,a=ALPHA):
    N=len(s);o=np.argsort(s);sw=w[o]*(N/(N+1))/w.sum();cw=np.cumsum(sw);i=int(np.searchsorted(cw,1-a,side='left'));return np.inf if i>=N else s[o][i]
def load(f):
    z=np.load(f,allow_pickle=True);return z['mean'].astype(float),(z['variance'].astype(float) if 'variance' in z.files else None),z['labels'].astype(float),z['masks'].astype(bool),z['ids']
def paths(bb,method,d,seed):
    sp='scaffold';nm='Ensembles-pooled' if method=='Ensembles' else f'{method}-{seed}'
    if method=='Laplace':val=V5/'preds'/f'{bb}-Laplace'/sp/d/f'{nm}.npz';tst=V5/'preds_test'/f'{bb}-Laplace'/sp/d/f'{nm}.npz'
    else:
        val=V5/'preds'/bb/sp/d/f'{nm}.npz';tst=(V8/'results'/'GIN'/sp/d/f'{nm}.npz') if bb=='GIN' else (V4/'results'/bb/sp/d/f'{nm}.npz')
    return val,tst
def units(bb,methods=('Deterministic','Laplace','Ensembles')):
    for m in methods:
        for s in (['pooled'] if m=='Ensembles' else [0,100,200]):yield m,s
_smiles={}
def smiles_of(d,ids):
    if d not in _smiles:_smiles[d]={p:[r['smiles'] for r in csv.DictReader((V4/'data'/d/f'{p}.csv').open())] for p in ['train','valid','test']}
    out=[]
    for i in ids:
        _,part,row=str(i).split('/');out.append(_smiles[d][part][int(row)])
    return out
def groups(d,ids):
    g=[]
    for k,s in enumerate(smiles_of(d,ids)):
        m=Chem.MolFromSmiles(s);g.append(f'unparsable{k}' if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    return np.array(g)
def cls_cov(p,y,q):
    in1=(1-p)<=q;in0=p<=q;return float(np.where(y==1,in1,in0).mean()),float((in1.astype(int)+in0).mean())
def reg_cov(m,v,y,q):
    sd=np.sqrt(v);e=np.abs(y-m);cov=float((e<=q*sd).mean());w=2*np.asarray(q)*sd;return cov,(float(w.mean()) if np.isfinite(w).all() else float('nan'))
def cls_scores(p,y):return np.where(y==1,1-p,p)
def reg_scores(m,v,y):return np.abs(y-m)/np.sqrt(v)
