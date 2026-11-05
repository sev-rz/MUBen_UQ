"""Bridge check (PROTOCOL.md): authors' unmodified Uni-Mol trainer (authors_out) vs our Deterministic Uni-Mol on the SAME seeds, test metrics. ESOL and BACE seeds 0-9, QM7 seeds 0-4.
Welch difference of means in units of the pooled seed SD: consistent if |d| < 1; different if |d| >= 1 and the 95 % interval excludes 0; otherwise unclear."""
import os,sys,glob,json
from pathlib import Path
V10=Path(__file__).resolve().parent;sys.path.insert(0,str(V10));os.environ['TIERB_OUT']=str(V10)
import numpy as np,pandas as pd
from scipy import stats
sys.path.insert(0,str(V10.parent/'repos'/'MUBen'))
from muben.utils.io import load_results
from muben.utils.metrics import regression_metrics, classification_metrics
import tierb_unimol as u
import tierb_eval as ev
t=u.t
PLAN={'esol':range(10),'bace':range(10),'qm7':range(5)}
def authors(d,seeds):
    rows=[]
    for s in seeds:
        p=glob.glob(str(V10/'authors_out'/d/f'seed-{s}'/'out'/d/'Uni-Mol'/'none'/f'seed-{s}'/'preds'/'*.pt'))
        if not p:continue
        preds,var,lbs,masks=load_results(p);m=regression_metrics(preds,var,lbs,masks) if var is not None else classification_metrics(preds,lbs,masks)
        rows.append({'seed':s,**{k:float(v['macro-avg']) for k,v in m.items()}})
    return pd.DataFrame(rows)
def ours(d,seeds):
    c=u.UCtx('scaffold',d);rows=[]
    for s in seeds:
        net,_=u.fit(c,'det',s);mu,var=ev.convert(c,u.agg(u.predict_raw(net,c,'test'),c.reg));m=ev.macro(c,mu,var);rows.append({'seed':s,**m})
    return pd.DataFrame(rows)
out=[];REG={'esol','qm7'}
for d,seeds in PLAN.items():
    A=authors(d,seeds);O=ours(d,[int(x) for x in A.seed]);A=A[A.seed.isin(O.seed)]
    for m in (['rmse','mae','nll','ce'] if d in REG else ['roc-auc','prc-auc','ece','nll','brier']):
        a=A[m].astype(float).values;b=O[m].astype(float).values;na,nb=len(a),len(b);sd=np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2));diff=a.mean()-b.mean()
        se=np.sqrt(a.var(ddof=1)/na+b.var(ddof=1)/nb);df=se**4/((a.var(ddof=1)/na)**2/(na-1)+(b.var(ddof=1)/nb)**2/(nb-1));h=stats.t.ppf(.975,df)*se;d_=diff/sd if sd>0 else np.nan
        cls='consistent' if abs(d_)<1 else ('different' if (diff-h>0 or diff+h<0) else 'unclear')
        out.append({'dataset':d,'metric':m,'authors':round(a.mean(),4),'ours':round(b.mean(),4),'n':f'{na}/{nb}','d':round(d_,2),'ci95_lo':round(diff-h,4),'ci95_hi':round(diff+h,4),'class':cls})
R=pd.DataFrame(out);R.to_csv(V10/'results'/'bridge_check.csv',index=False);pd.set_option('display.width',200);print(R.to_string(index=False));print(R['class'].value_counts().to_dict())
