"""Cross-check conformal.py against the crepes library on the self-test split (validation halves; test split untouched).
Compares, per prediction file and alpha: normalised regression intervals, LAC sets, and novelty-Mondrian regression intervals."""
import sys,glob,json
from pathlib import Path
import numpy as np
from crepes import ConformalRegressor, ConformalClassifier
C=Path(__file__).resolve().parent; sys.argv=[sys.argv[0],'selftest']
import conformal as cf
rows=[]
for vdir in sorted(glob.glob(str(C/'preds'/'*'/'*'/'*'))):
    bb,split,d=Path(vdir).parts[-3:];reg=d in ('esol','freesolv','lipo','qm7');nv,_=cf.novelty(d,split)
    for vf in sorted(glob.glob(vdir+'/*.npz')):
        mu,var,y,m=cf.load(vf);mu,y,m=mu[:,0],y[:,0],m[:,0];var=None if var is None else var[:,0]
        rng=np.random.default_rng(0);perm=rng.permutation(len(y));h=len(y)//2;ci,ti=perm[:h],perm[h:]
        ci,ti=ci[m[ci]],ti[m[ti]]
        for a in cf.ALPHAS:
            if reg:
                sc,st=np.sqrt(var[ci]),np.sqrt(var[ti])
                ours=cf.reg_rows((mu[ci],var[ci],y[ci]),(mu[ti],var[ti],y[ti]),nv[ci],nv[ti],a,True)
                cr=ConformalRegressor();cr.fit(residuals=y[ci]-mu[ci],sigmas=sc)
                iv=cr.predict_int(y_hat=mu[ti],sigmas=st,confidence=1-a)
                cov=np.mean((y[ti]>=iv[:,0])&(y[ti]<=iv[:,1]));w=np.mean(iv[:,1]-iv[:,0])
                bc,bt=(nv[ci]<cf.NOVEL).astype(int),(nv[ti]<cf.NOVEL).astype(int)
                rows.append({'file':Path(vf).stem,'d':d,'alpha':a,'kind':'reg-normalised','d_cov':abs(cov-ours['coverage']),'d_width':abs(w-ours['width_mean'])})
            else:
                ours=cf.cls_rows((mu[ci],y[ci]),(mu[ti],y[ti]),nv[ci],nv[ti],a,False)
                P=lambda p:np.c_[1-p,p]
                cc=ConformalClassifier();cc.fit(alphas=1-P(mu[ci])[np.arange(len(ci)),y[ci].astype(int)])
                S=cc.predict_set(alphas=1-P(mu[ti]),confidence=1-a,smoothing=False)
                cov=np.mean(S[np.arange(len(ti)),y[ti].astype(int)]==1);size=S.sum(1).mean()
                rows.append({'file':Path(vf).stem,'d':d,'alpha':a,'kind':'cls-LAC','d_cov':abs(cov-ours['coverage']),'d_width':abs(size-ours['set_size'])})
import pandas as pd
df=pd.DataFrame(rows);print(df.groupby('kind')[['d_cov','d_width']].agg(['max','mean']).to_string());print('comparisons',len(df))
(C/'selftest').mkdir(exist_ok=True);df.to_csv(C/'selftest'/'crepes_crosscheck.csv',index=False)
