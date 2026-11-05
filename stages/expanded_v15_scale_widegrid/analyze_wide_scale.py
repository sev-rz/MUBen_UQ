"""Analysis of items B1/B2 (PROTOCOL.md): R10 verdicts for Laplace vs Deterministic (primary and wide grid) and Laplace vs one-parameter recalibration; sign-flip family tests.  python analyze_wide_scale.py"""
import json,itertools,numpy as np,pandas as pd
from pathlib import Path
from scipy import stats
D=Path(__file__).resolve().parent;DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7']
def verdict(x,pos='improves',neg='worsens'):
    x=np.asarray(x,float);n=len(x);m=x.mean();se=x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;t=stats.t.ppf(0.975,n-1);lo,hi=m-t*se,m+t*se
    if (x<0).mean()>=0.9 and hi<0:v=pos
    elif (x>0).mean()>=0.9 and lo>0:v=neg
    else:v='unclear'
    return v,m,lo,hi,(x<0).mean(),(m/se if se>0 else 0.)
def signflip(z):
    z=np.asarray(z,float);obs=z.sum();cnt=0;tot=0
    for s in itertools.product([1,-1],repeat=len(z)):tot+=1;cnt+=(np.dot(s,z)<=obs+1e-12)
    return cnt/tot
rows=[]
for bb in ['DNN','ChemBERTa','GIN','Uni-Mol']:
    for d in DS:
        f=D/'results'/bb/f'{d}.json'
        if not f.exists():continue
        r=json.loads(f.read_text());g=lambda k:np.array([x[k]['nll'] for x in r])
        L,Wd,S,Dt=g('laplace'),g('laplace_wide'),g('scaling'),g('deterministic')
        vLD=verdict(L-Dt);vWD=verdict(Wd-Dt);vSD=verdict(S-Dt);vLS=verdict(L-S,'Laplace better','scaling better');vWL=verdict(Wd-L)
        rows.append({'backbone':bb,'dataset':d,'n':len(r),'mean_LD':vLD[1],'mean_SD':vSD[1],'mean_LS':vLS[1],'mean_WD':vWD[1],'mean_WL':vWL[1],'frac_L_better_than_S':vLS[4],
                     'verdict_LD':vLD[0],'verdict_WD':vWD[0],'verdict_SD':vSD[0],'verdict_LS':vLS[0],'t_LS':vLS[5],'t_LD':vLD[5],'edge_primary':sum(x['edge_primary'] for x in r),'edge_wide':sum(x['edge_wide'] for x in r),
                     'task_fits':sum(x['tasks'] for x in r),'max_repro_diff':max((x['repro_diff_vs_itemB'] or 0) for x in r)})
A=pd.DataFrame(rows);A.to_csv(D/'wide_scale_by_dataset.csv',index=False)
F=[]
for bb,g in A.groupby('backbone'):
    if len(g)<8:continue
    F.append({'backbone':bb,'datasets':len(g),'LD_improves':int((g.verdict_LD=='improves').sum()),'WD_improves':int((g.verdict_WD=='improves').sum()),'verdict_unchanged_wide':int((g.verdict_LD==g.verdict_WD).sum()),
              'SD_improves':int((g.verdict_SD=='improves').sum()),'Laplace_better_than_scaling':int((g.verdict_LS=='Laplace better').sum()),'scaling_better':int((g.verdict_LS=='scaling better').sum()),'unclear_LS':int((g.verdict_LS=='unclear').sum()),
              'mean_LD':g.mean_LD.mean(),'mean_SD':g.mean_SD.mean(),'mean_LS':g.mean_LS.mean(),'mean_WL':g.mean_WL.mean(),'p_signflip_LS_laplace_better':signflip(g.t_LS.values),'edge_primary_share':g.edge_primary.sum()/g.task_fits.sum(),'edge_wide_share':g.edge_wide.sum()/g.task_fits.sum(),'max_repro_diff':g.max_repro_diff.max()})
F=pd.DataFrame(F);F.to_csv(D/'wide_scale_family.csv',index=False)
pd.set_option('display.width',260);pd.set_option('display.max_columns',40)
print(A[['backbone','dataset','n','mean_LD','mean_SD','mean_LS','frac_L_better_than_S','verdict_LD','verdict_WD','verdict_SD','verdict_LS','mean_WL','edge_primary','edge_wide']].round(3).to_string(index=False))
print();print(F.round(4).to_string(index=False))
