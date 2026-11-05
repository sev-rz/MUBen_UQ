"""Check E (n0096): bootstrap intervals and a family-level sign-flip test for the paired NLL difference Laplace minus Deterministic, on the SAVED test predictions.
Per-molecule NLL must first reproduce the stored metric. Bootstrap: 2000 resamples of test molecules (jointly across tasks); indices shared across seeds on scaffold (same test set), independent per split on random.
Dataset statistic z = mean difference / bootstrap SE. Family (per backbone and split type, 6 tests): exact sign-flip test over the 8 dataset z values, one-sided for improvement, Holm correction over the 6 tests."""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V4=W/'expanded_v4'/'results';V6=W/'expanded_v6_laplace';V8=W/'expanded_v8_gin'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'}
DET={'DNN':lambda s,d:V4/'DNN'/s/d,'ChemBERTa':lambda s,d:V4/'ChemBERTa'/s/d,'GIN':lambda s,d:V8/'results'/'GIN'/s/d}
LAP={'DNN':lambda s,d:V6/'results'/s/d,'ChemBERTa':lambda s,d:V6/'chemberta'/'results'/s/d,'GIN':lambda s,d:V8/'laplace'/'results'/s/d}
SPLITS={'scaffold':('scaffold',[0,100,200]),'random-0':('random',[0]),'random-1':('random',[100]),'random-2':('random',[200])}
B=2000;rng=np.random.default_rng(20261011)
def per_mol_nll(z,reg):
    y=z['labels'].astype(float);m=z['masks'].astype(bool);mu=z['mean'].astype(float)
    if reg:
        v=z['variance'].astype(float);return 0.5*(np.log(v)+(y-mu)**2/v),m
    with np.errstate(divide='ignore'):lp=np.maximum(np.log(mu),-100.0);l1=np.maximum(np.log(1-mu),-100.0)   # PyTorch binary_cross_entropy clamps each log at -100 (the stored definition)
    return -(y*lp+(1-y)*l1),m
def scored_tasks(z,reg):
    if reg:return np.ones(z['labels'].shape[1],bool)
    y=z['labels'].astype(float);m=z['masks'].astype(bool);ok=[]
    for j in range(y.shape[1]):
        yy=y[m[:,j],j];ok.append(0<yy.mean()<1)
    return np.array(ok)
def macro(nll,m,ok,idx=None):
    nll0=np.where(m,nll,0.0)
    if idx is None:return float(np.mean([nll0[:,j].sum()/m[:,j].sum() for j in range(nll.shape[1]) if ok[j]]))
    S=nll0[idx].sum(1);N=m[idx].sum(1);return np.mean([S[:,j]/N[:,j] for j in range(nll.shape[1]) if ok[j]],axis=0)
def stored(path,method,seed):
    rows=json.loads((path/'per_seed_metrics.json').read_text());r=[x for x in rows if x['method']==method and str(x['seed'])==str(seed) and x.get('subgroup','all')=='all'];return r[0]['nll']
rows=[];maxdiff=0.0
for bb in DET:
    for d in DS:
        reg=d in REG;res={'scaffold':[], 'random':[]}
        for sp,(st,seeds) in SPLITS.items():
            diffs=[]
            for s in seeds:
                zd=np.load(DET[bb](sp,d)/f'Deterministic-{s}.npz');zl=np.load(LAP[bb](sp,d)/f'Laplace-{s}.npz')
                nd,m=per_mol_nll(zd,reg);nl,m2=per_mol_nll(zl,reg);assert (m==m2).all();ok=scored_tasks(zd,reg)
                maxdiff=max(maxdiff,abs(macro(nd,m,ok)-stored(DET[bb](sp,d),'Deterministic',s)),abs(macro(nl,m,ok)-stored(LAP[bb](sp,d),'Laplace',s)))
                diffs.append((nl-nd,m,ok))
            n=diffs[0][0].shape[0]
            if st=='scaffold':
                idx=rng.integers(0,n,(B,n));boot=np.mean([macro(D,m,ok,idx) for D,m,ok in diffs],axis=0);est=np.mean([macro(D,m,ok) for D,m,ok in diffs])
                res['scaffold'].append((est,boot))
            else:
                idx=rng.integers(0,n,(B,n));D,m,ok=diffs[0];res['random'].append((macro(D,m,ok),macro(D,m,ok,idx)))
        # random splits: average over the 3 splits (independent test sets)
        for st in ['scaffold','random']:
            if st=='scaffold':est,boot=res['scaffold'][0]
            else:est=np.mean([e for e,_ in res['random']]);boot=np.mean([b for _,b in res['random']],axis=0)
            lo,hi=np.percentile(boot,[2.5,97.5]);se=float(boot.std());ident=abs(est)<1e-4 and se<1e-4   # identical predictions (prior precision at the upper grid edge): no effect, z = 0
            z=0.0 if ident else (est/se if se>1e-12 else 0.0)
            rows.append({'backbone':bb,'stype':st,'dataset':d,'diff':round(est,4),'lo':round(lo,4),'hi':round(hi,4),'se':round(se,4),'z':round(z,2),'verdict':'identical' if ident else ('improves' if hi<0 else ('worsens' if lo>0 else 'unclear'))})
print('max |recomputed NLL - stored NLL| over all Deterministic and Laplace groups:',maxdiff)
R=pd.DataFrame(rows);R.to_csv(W/'expanded_v9_checks'/'stats_E_dataset.csv',index=False)
fam=[]
for (bb,st),g in R.groupby(['backbone','stype']):
    z=g.set_index('dataset').loc[DS].z.to_numpy();obs=z.sum()
    flips=np.array([np.sum(z*np.array(s)) for s in itertools.product([1,-1],repeat=len(z))]);p=float((flips<=obs+1e-12).mean())
    fam.append({'backbone':bb,'stype':st,'sum_z':round(obs,2),'p_one_sided':round(p,4),'n_improves':int((g.verdict=='improves').sum()),'n_unclear':int((g.verdict=='unclear').sum()),'n_worsens':int((g.verdict=='worsens').sum()),'n_identical':int((g.verdict=='identical').sum())})
F=pd.DataFrame(fam).sort_values('p_one_sided').reset_index(drop=True);m=len(F)
F['holm_p']=[round(min(1.0,max(F.p_one_sided[:i+1]*np.array([m-k for k in range(i+1)]))),4) for i in range(m)]
F.to_csv(W/'expanded_v9_checks'/'stats_E_family.csv',index=False)
pd.set_option('display.width',200);print('\nDATASET LEVEL (95 % bootstrap interval of the paired NLL difference; negative = Laplace better)')
for st in ['scaffold','random']:
    print('\n',st);print(R[R.stype==st].pivot(index='dataset',columns='backbone',values='verdict').loc[DS].to_string())
print('\nFAMILY LEVEL (exact sign-flip test, Holm over 6)');print(F.to_string(index=False))
print('\nDATASET DETAIL');print(R.to_string(index=False))
