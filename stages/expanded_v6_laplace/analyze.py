"""Registered analysis of Laplace on Tier B (PROTOCOL.md section 5). Reads per_seed_metrics.json of expanded_v6_laplace and expanded_v4 (DNN). No tuning."""
import json, glob, numpy as np, pandas as pd
from pathlib import Path
W=Path('.');V6=W/'expanded_v6_laplace';V4=W/'expanded_v4'/'results'/'DNN'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];SPL=['scaffold','random-0','random-1','random-2']
L=[];B=[]
for sp in SPL:
    for d in DS:
        L+=json.loads((V6/'results'/sp/d/'per_seed_metrics.json').read_text());B+=json.loads((V4/sp/d/'per_seed_metrics.json').read_text())
L=pd.DataFrame(L);B=pd.DataFrame(B);L=L[L.subgroup=='all'].copy()
for df in (L,B):df['stype']=np.where(df.split=='scaffold','scaffold','random')
def get(df,m,metric):return df[df.method==m].groupby(['stype','dataset','split','seed'],as_index=False)[metric].mean()
def paired(a,b,metric='nll'):
    x=get(L,a,metric);y=get(B if b not in L.method.unique() else L,b,metric)
    if (y.seed.astype(str)=='pooled').any():   # scaffold ensembles: one pooled value per dataset
        yy=y.groupby(['stype','dataset','split'],as_index=False)[metric].mean().rename(columns={metric:'y'});m=x.merge(yy,on=['stype','dataset','split'])
    else:m=x.merge(y.rename(columns={metric:'y'}),on=['stype','dataset','split','seed'])
    m['delta']=m[metric]-m['y'];return m
def summarize(m):
    g=m.groupby(['stype','dataset']).delta.agg(['mean','std','count',lambda s:(s<0).all(),lambda s:(s>0).all()]).reset_index()
    g.columns=['stype','dataset','mean_delta','sd','repeats','all_neg','all_pos'];g['verdict']=np.where(g.all_neg,'improves',np.where(g.all_pos,'worsens','mixed'));return g
pd.set_option('display.width',200)
out={}
for name,(a,b) in {'primary: Laplace - Deterministic (NLL)':('Laplace','Deterministic'),'Laplace - MC dropout (NLL)':('Laplace','MC dropout'),'Laplace - Ensembles (NLL)':('Laplace','Ensembles')}.items():
    s=summarize(paired(a,b));out[name]=s;print('\n###',name);print(s.pivot(index='dataset',columns='stype',values='mean_delta').reindex(DS).round(4).to_string())
    for st in ['scaffold','random']:
        v=s[s.stype==st].verdict.value_counts().to_dict();print(f'   {st}: {v}  (rule: overall claim needs >=6 of 8 datasets with the same all-repeats sign)')
s.to_csv(V6/'results'/'paired_vs_ensembles.csv',index=False);out['primary: Laplace - Deterministic (NLL)'].to_csv(V6/'results'/'primary_laplace_minus_deterministic_nll.csv',index=False)
# secondary metrics, classification and regression
print('\n### Other metrics, Laplace - Deterministic, mean over repeats')
for metric in ['roc-auc','ece','brier','rmse','mae','ce']:
    m=paired('Laplace','Deterministic',metric)
    if len(m):print(metric.ljust(8),m.groupby('stype').delta.mean().round(4).to_dict(),'| datasets:',sorted(m.dataset.unique()))
# novelty variants (scaffold conclusions only)
print('\n### Novelty arm, scaffold: variant NLL minus plain Laplace NLL')
rows=[]
for v in ['Laplace constant','Novelty only','Novelty Laplace']:
    m=paired(v,'Laplace');m=m[m.stype=='scaffold'];s=summarize(m);s['variant']=v;rows.append(s)
nv=pd.concat(rows);print(nv.pivot(index='dataset',columns='variant',values='mean_delta').reindex(DS).round(4).to_string())
for v in nv.variant.unique():print('  ',v,nv[nv.variant==v].verdict.value_counts().to_dict())
nv.to_csv(V6/'results'/'novelty_vs_laplace_scaffold.csv',index=False)
# novelty subgroup NLL on scaffold (Laplace vs Novelty Laplace), pooled over seeds
A=pd.DataFrame([r for sp in ['scaffold'] for d in DS for r in json.loads((V6/'results'/sp/d/'per_seed_metrics.json').read_text())])
sub=A[(A.subgroup!='all')&(A.method.isin(['Laplace','Novelty Laplace']))].groupby(['dataset','subgroup','method'],as_index=False).agg(nll=('nll','mean'),n=('n','first'))
print('\n### Scaffold subgroup NLL (novel = similarity<0.4):');print(sub.pivot_table(index=['dataset','subgroup'],columns='method',values='nll').round(3).join(sub.groupby(['dataset','subgroup']).n.first()).to_string())
# cost: fit seconds vs ensemble members
fs=[json.loads(Path(p).read_text())['fit_seconds'] for p in glob.glob(str(V6/'results'/'*'/'*'/'seed-*'/'fit.json'))]
print(f'\n### Cost: Laplace fit seconds per net: median {np.median(fs):.2f}, max {max(fs):.2f}, n={len(fs)} (ensemble = 10 nets trained on scaffold, 3 on random)')
