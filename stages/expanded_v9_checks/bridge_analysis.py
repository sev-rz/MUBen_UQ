"""Checks A and B of the registered fix plan (n0095). A: authors' GIN trainer (seeds 0-9; tox21 0-4) vs OUR saved GIN Deterministic nets (scaffold ensemble members, seeds 0-9) evaluated on test with the frozen code.
B: authors' ChemBERTa trainer (ESOL, QM7; seeds 0-2) vs our Tier-B ChemBERTa Deterministic. Metrics of the authors' runs come from their own functions."""
import sys, os, glob, json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V4=W/'expanded_v4';V8=W/'expanded_v8_gin';REPO=W/'repos'/'MUBen'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(REPO)]
from muben.utils.io import load_results
from muben.utils.metrics import regression_metrics, classification_metrics
sys.path.insert(0,str(V8))
import tierb_gin as g
t,e=g.t,g.e
import tierb_eval as ev
REG=['esol','freesolv','lipo','qm7'];CLS=['bace','bbbp','clintox','tox21']
def authors_metrics(root,tag,ds,seeds):
    rows=[]
    for s in seeds:
        p=glob.glob(f'{root}/{ds}/{tag}/none/seed-{s}/preds/*.pt')
        if not p:continue
        preds,var,lbs,masks=load_results(p)
        m=regression_metrics(preds,var,lbs,masks) if var is not None else classification_metrics(preds,lbs,masks)
        rows.append({'seed':s,**{k:float(v['macro-avg']) for k,v in m.items()}})
    return pd.DataFrame(rows)
def ours_gin(ds,seeds):
    c=g.GCtx('scaffold',ds);rows=[]
    for s in seeds:
        net,_=g.fit(c,'det',s);o=g.predict(net,c.graphs['test']);mu,var=ev.convert(c,o);m=ev.macro(c,mu,var);rows.append({'seed':s,**m})
    return pd.DataFrame(rows)
def compare(a,b,metrics):
    out=[]
    for m in metrics:
        if m not in a or m not in b:continue
        x,y=a[m].astype(float).to_numpy(),b[m].astype(float).to_numpy();n1,n2=len(x),len(y);s1,s2=x.std(ddof=1),y.std(ddof=1);diff=x.mean()-y.mean();pooled=np.sqrt((s1**2+s2**2)/2)
        se=np.sqrt(s1**2/n1+s2**2/n2);df=se**4/((s1**2/n1)**2/(n1-1)+(s2**2/n2)**2/(n2-1)) if se>0 else 1;ci=stats.t.interval(0.95,df,loc=diff,scale=se) if se>0 else (diff,diff)
        d=diff/pooled if pooled>0 else np.nan
        out.append({'metric':m,'authors':round(x.mean(),4),'ours':round(y.mean(),4),'n_authors':n1,'n_ours':n2,'diff':round(diff,4),'pooled_sd':round(pooled,4),'d':round(d,2),'ci95_lo':round(ci[0],4),'ci95_hi':round(ci[1],4),
                    'class':'different' if (abs(d)>=1 and not (ci[0]<=0<=ci[1])) else ('consistent' if abs(d)<1 else 'unclear')})
    return pd.DataFrame(out)
res=[]
for ds in REG+CLS:
    n=5 if ds=='tox21' else 10;a=authors_metrics(V8.parent/'expanded_v9_checks'/'gin_authors10_out','GIN',ds,range(n));b=ours_gin(ds,range(n))
    mets=['rmse','mae','nll','ce'] if ds in REG else ['roc-auc','prc-auc','ece','nll','brier']
    if ds in CLS:a=a.rename(columns={k:k for k in a.columns})
    c=compare(a,b,mets);c.insert(0,'dataset',ds);res.append(c);print(ds,'authors cols:',[x for x in a.columns if x!='seed'][:8],flush=True)
A=pd.concat(res);A.to_csv(W/'expanded_v9_checks'/'gin_bridge10.csv',index=False)
pd.set_option('display.width',220);print('\n=== A: GIN, authors trainer vs our pipeline, scaffold Deterministic');print(A.to_string(index=False))
print('\nclass counts:',A['class'].value_counts().to_dict())
# B: ChemBERTa bridge, rule |diff| < 2 x larger seed SD
rowsB=[]
for ds in ['esol','qm7']:
    a=authors_metrics(W/'expanded_v9_checks'/'chemberta_authors_out','ChemBERTa',ds,range(3))
    o=pd.DataFrame(json.loads((V4/'results'/'ChemBERTa'/'scaffold'/ds/'per_seed_metrics.json').read_text()));o=o[o.method=='Deterministic']
    for m in ['rmse','mae','nll','ce']:
        x,y=a[m].astype(float).to_numpy(),o[m].astype(float).to_numpy();sd=max(x.std(ddof=0),y.std(ddof=0));diff=abs(x.mean()-y.mean())
        rowsB.append({'dataset':ds,'metric':m,'authors_trainer':round(x.mean(),4),'ours':round(y.mean(),4),'abs_diff':round(diff,4),'max_seed_sd':round(sd,4),'ratio':round(diff/sd,2) if sd>0 else np.nan,'consistent_lt_2sd':bool(diff<2*sd)})
B=pd.DataFrame(rowsB);B.to_csv(W/'expanded_v9_checks'/'chemberta_bridge.csv',index=False);print('\n=== B: ChemBERTa bridge');print(B.to_string(index=False))
