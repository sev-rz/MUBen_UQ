import json, numpy as np, pandas as pd, sys
from pathlib import Path
sys.argv=['x'];exec(open('stats_E.py').read().split("rows=[];maxdiff=0.0")[0])
bad=[]
for bb in DET:
    for d in DS:
        reg=d in REG
        for sp,(st,seeds) in SPLITS.items():
            for s in seeds:
                for meth,loc in [('Deterministic',DET[bb](sp,d)),('Laplace',LAP[bb](sp,d))]:
                    z=np.load(loc/f'{meth}-{s}.npz');nl,m=per_mol_nll(z,reg);ok=scored_tasks(z,reg)
                    mine=macro(nl,m,ok);st_=stored(loc,meth,s);df=abs(mine-st_)
                    if df>1e-6:bad.append((bb,d,sp,s,meth,round(mine,5),round(st_,5),round(df,5),int(ok.sum()),len(ok)))
B=pd.DataFrame(bad,columns=['backbone','dataset','split','seed','method','mine','stored','absdiff','tasks_scored','tasks'])
print('groups with |diff| > 1e-6:',len(B));print(B.groupby(['backbone','dataset']).size().to_string());print(B.sort_values('absdiff',ascending=False).head(12).to_string(index=False))
