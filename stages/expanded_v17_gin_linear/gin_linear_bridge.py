"""B3 runner (PROTOCOL.md): our GIN Deterministic with the linear schedule vs the authors' ten-seed bridge runs (classification datasets), same comparison rule as gin_bridge10.csv.
  python gin_linear_bridge.py"""
import sys,os,glob,json,numpy as np,pandas as pd
from pathlib import Path
from scipy import stats
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V17=W/'expanded_v17_gin_linear';V9=W/'expanded_v9_checks';REPO=W/'repos'/'MUBen'
os.environ['TIERB_OUT']=str(V17);sys.path[:0]=[str(REPO),str(V17)]
from muben.utils.io import load_results
from muben.utils.metrics import classification_metrics
import tierb_gin_linear as g
import tierb_eval as ev
CLS=['bace','bbbp','clintox','tox21']
def authors_metrics(ds,seeds):
    rows=[]
    for s in seeds:
        p=glob.glob(f'{V9}/gin_authors10_out/{ds}/GIN/none/seed-{s}/preds/*.pt')
        if not p:continue
        preds,var,lbs,masks=load_results(p);m=classification_metrics(preds,lbs,masks);rows.append({'seed':s,**{k:float(v['macro-avg']) for k,v in m.items()}})
    return pd.DataFrame(rows)
def ours_linear(ds,seeds):
    c=g.GCtx('scaffold',ds);rows=[]
    for s in seeds:
        net,_=g.fit(c,'det',s);o=g.predict(net,c.graphs['test']);mu,var=ev.convert(c,o);m=ev.macro(c,mu,var);rows.append({'seed':s,**m});print('fit',ds,s,round(m['roc-auc'],4),flush=True)
    return pd.DataFrame(rows)
def compare(a,b,metrics):
    out=[]
    for m in metrics:
        x,y=a[m].astype(float).to_numpy(),b[m].astype(float).to_numpy();n1,n2=len(x),len(y);s1,s2=x.std(ddof=1),y.std(ddof=1);diff=x.mean()-y.mean();pooled=np.sqrt((s1**2+s2**2)/2)
        se=np.sqrt(s1**2/n1+s2**2/n2);df=se**4/((s1**2/n1)**2/(n1-1)+(s2**2/n2)**2/(n2-1)) if se>0 else 1;ci=stats.t.interval(0.95,df,loc=diff,scale=se) if se>0 else (diff,diff)
        d=diff/pooled if pooled>0 else np.nan
        out.append({'metric':m,'authors':round(x.mean(),4),'ours_linear':round(y.mean(),4),'n_authors':n1,'n_ours':n2,'diff':round(diff,4),'pooled_sd':round(pooled,4),'d':round(d,2),'ci95_lo':round(ci[0],4),'ci95_hi':round(ci[1],4),
                    'class':'different' if (abs(d)>=1 and not (ci[0]<=0<=ci[1])) else ('consistent' if abs(d)<1 else 'unclear')})
    return pd.DataFrame(out)
res=[];per=[]
for ds in CLS:
    n=5 if ds=='tox21' else 10;a=authors_metrics(ds,range(n));b=ours_linear(ds,range(n));b.insert(0,'dataset',ds);per.append(b)
    c=compare(a,b,['roc-auc','prc-auc','ece','nll','brier']);c.insert(0,'dataset',ds);res.append(c)
A=pd.concat(res);old=pd.read_csv(V9/'gin_bridge10.csv')[['dataset','metric','ours','d','class']].rename(columns={'ours':'ours_constant','d':'d_constant','class':'class_constant'})
A=A.merge(old,on=['dataset','metric'],how='left');A.to_csv(V17/'gin_linear_bridge.csv',index=False);pd.concat(per).to_csv(V17/'gin_linear_per_seed.csv',index=False)
pd.set_option('display.width',250);print(A.to_string(index=False))
print('\nclass counts linear :',A['class'].value_counts().to_dict());print('class counts constant:',A['class_constant'].value_counts().to_dict());print('DONE')
