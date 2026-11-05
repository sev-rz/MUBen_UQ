"""Compare ChemBERTa arms (ESOL, QM7; Deterministic, scaffold) and the GIN head-init arm. Authors' metrics from their own functions (load_results/regression_metrics)."""
import sys,json,glob,numpy as np,pandas as pd
sys.path.insert(0,'/l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v9_checks')
import bridge_analysis as b
W=b.W;V12=W/'expanded_v12_headinit'
def ours_tierb(d):
    r=pd.DataFrame(json.loads((W/'expanded_v4'/'results'/'ChemBERTa'/'scaffold'/d/'per_seed_metrics.json').read_text()));return r[r.method=='Deterministic']
def arm(a,d):return pd.DataFrame(json.loads((V12/'metrics'/f'ChemBERTa_{a}_{d}.json').read_text()))
rows=[]
for d in ['esol','qm7']:
    sets={'authors, linear sched (earlier bridge)':b.authors_metrics(str(W/'expanded_v9_checks'/'chemberta_authors_out'),'ChemBERTa',d,[0,1,2]),
          'authors, constant sched (default)':b.authors_metrics(str(W/'expanded_v11_chemberta_sched'/'authors_const_out'),'ChemBERTa',d,[0,1,2]),
          'ours Tier-B (constant, Xavier head)':ours_tierb(d),'ours + linear sched':arm('linear',d),'ours + default head init':arm('default',d)}
    for k,df in sets.items():
        r={'dataset':d,'arm':k,'n':len(df)}
        for m in ['rmse','mae','nll','ce']:r[m]=f"{df[m].mean():.4f} ± {df[m].std(ddof=0):.4f}"
        rows.append(r)
R=pd.DataFrame(rows);pd.set_option('display.width',250);print(R.to_string(index=False));R.to_csv(V12/'chemberta_arms.csv',index=False)
# GIN head-init arm vs authors' trainer (10 seeds) and vs ours Xavier (10 seeds)
g=[]
for d in ['bace','clintox']:
    ai=b.authors_metrics(str(W/'expanded_v9_checks'/'gin_authors10_out'),'GIN',d,range(10));hi=pd.DataFrame(json.loads((V12/'metrics'/f'GIN_{d}.json').read_text()))
    xa=b.ours_gin(d,range(10)) if False else None
    for m in ['roc-auc','prc-auc','ece','nll','brier']:
        a=ai[m].astype(float);h=hi[m].astype(float);sd=np.sqrt((a.var(ddof=1)+h.var(ddof=1))/2);g.append({'dataset':d,'metric':m,'authors(10)':f'{a.mean():.4f}','ours default-init':f'{h.mean():.4f} (n={len(h)})','d_default_vs_authors':round((h.mean()-a.mean())/sd,2)})
print(pd.DataFrame(g).to_string(index=False));pd.DataFrame(g).to_csv(V12/'gin_headinit_vs_authors.csv',index=False)
