"""Uni-Mol copy of conformal_v2.py (same analysis; families Uni-Mol and Uni-Mol-Laplace only; Uni-Mol test paths added; outputs results_unimol / selftest_unimol).
Conformal analysis (v2 of conformal.py: identical analysis as frozen in PROTOCOL.md; only change = unparsable-SMILES guard in novelty()). Saved predictions only.
  python conformal.py run       -> results/conformal_{regression,classification}.csv (needs Tier B complete)
  python conformal.py selftest  -> same code on validation halves only (test split untouched)"""
import sys,os,json,math,glob
from pathlib import Path
import numpy as np, pandas as pd
C=Path(__file__).resolve().parent; V=C.parent/'expanded_v4'; sys.path.insert(0,str(V)); os.chdir(V)
import tierb_dnn as t
from rdkit import Chem, DataStructs, RDLogger; from rdkit.Chem import AllChem; RDLogger.DisableLog('rdApp.*')
ALPHAS=[0.1,0.2]; NOVEL=0.4; MIN_CAL=20

def smiles(d):
    import csv;return [r['smiles'] for p in ['train','valid','test'] for r in csv.DictReader((t.ROOT/'data'/d/f'{p}.csv').open())]
def novelty(d,split):
    f=C/'cache'/f'novelty_{split}_{d}.npz'
    if f.exists():z=np.load(f);return z['valid'],z['test']
    sm=smiles(d);ii=t.split_index(d,split)
    def fp(s):   # v2: unparsable SMILES (8 Tox21 molecules under the current RDKit) -> no fingerprint
        m=Chem.MolFromSmiles(s);return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048,useChirality=False)
    tr=[x for x in (fp(sm[i]) for i in ii['train']) if x is not None]
    mx=lambda idx:np.array([0.0 if (q:=fp(sm[i])) is None else max(DataStructs.BulkTanimotoSimilarity(q,tr)) for i in idx])   # unparsable query: similarity 0 = maximally novel, as in stage 1 (D5)
    v,te=mx(ii['valid']),mx(ii['test']);f.parent.mkdir(exist_ok=True);np.savez(f,valid=v,test=te);return v,te
def qhat(s,a):
    s=np.sort(np.asarray(s));n=len(s);k=math.ceil((n+1)*(1-a));return np.inf if k>n else s[k-1]
def wilson(x,n,z=1.96):
    if n==0:return (np.nan,np.nan)
    p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
def verdict(cov,n,a):
    lo,hi=wilson(round(cov*n),n);return 'held' if lo<=1-a<=hi else ('under' if hi<1-a else 'over')

def reg_rows(cal,tst,nv_c,nv_t,a,sig):
    """cal/tst: (mu,var,y). sig: normalised (True) or unnormalised baseline."""
    (mc,vc,yc),(mt,vt,yt)=cal,tst
    sc=np.sqrt(vc) if sig else np.ones_like(mc);st=np.sqrt(vt) if sig else np.ones_like(mt)
    s=np.abs(yc-mc)/sc;q=qhat(s,a);w=2*q*st;hit=np.abs(yt-mt)<=q*st
    nov_t=nv_t<NOVEL;nov_c=nv_c<NOVEL
    r={'coverage':hit.mean(),'width_mean':w.mean(),'width_median':np.median(w),'n_cal':len(s),'n_test':len(hit),
       'cov_novel':hit[nov_t].mean() if nov_t.any() else np.nan,'cov_familiar':hit[~nov_t].mean() if (~nov_t).any() else np.nan,
       'n_novel_test':int(nov_t.sum()),'n_novel_cal':int(nov_c.sum())}
    m=np.full(len(hit),np.nan);wm=np.full(len(hit),np.nan)
    for g_c,g_t,name in [(nov_c,nov_t,'novel'),(~nov_c,~nov_t,'familiar')]:
        if g_c.sum()>=MIN_CAL and g_t.any():
            qg=qhat(s[g_c],a);m[g_t]=np.abs(yt[g_t]-mt[g_t])<=qg*st[g_t];wm[g_t]=2*qg*st[g_t]
            r[f'mondrian_cov_{name}']=float(np.nanmean(m[g_t]));r[f'mondrian_width_{name}']=float(np.nanmean(wm[g_t]))
        else:r[f'mondrian_cov_{name}']=np.nan;r[f'mondrian_width_{name}']=np.nan
    r['verdict']=verdict(r['coverage'],len(hit),a);return r

def cls_rows(cal,tst,nv_c,nv_t,a,classcond):
    (pc,yc),(pt,yt)=cal,tst
    sc=np.where(yc==1,1-pc,pc)           # LAC score = 1 - p(true class)
    if classcond:q={k:qhat(sc[yc==k],a) for k in (0,1)}
    else:qq=qhat(sc,a);q={0:qq,1:qq}
    in1=(1-pt)<=q[1];in0=pt<=q[0];size=in1.astype(int)+in0;hit=np.where(yt==1,in1,in0)
    single=size==1;pred1=in1&~in0
    nov_t=nv_t<NOVEL;nov_c=nv_c<NOVEL
    r={'coverage':hit.mean(),'set_size':size.mean(),'singleton':single.mean(),'empty':(size==0).mean(),
       'singleton_acc':(pred1[single]==(yt[single]==1)).mean() if single.any() else np.nan,'n_cal':len(sc),'n_test':len(hit),
       'cov_novel':hit[nov_t].mean() if nov_t.any() else np.nan,'cov_familiar':hit[~nov_t].mean() if (~nov_t).any() else np.nan,
       'n_novel_test':int(nov_t.sum()),'n_novel_cal':int(nov_c.sum())}
    if not classcond:
        for g_c,g_t,name in [(nov_c,nov_t,'novel'),(~nov_c,~nov_t,'familiar')]:
            if g_c.sum()>=MIN_CAL and g_t.any():
                qg=qhat(sc[g_c],a);h=np.where(yt[g_t]==1,(1-pt[g_t])<=qg,pt[g_t]<=qg)
                r[f'mondrian_cov_{name}']=h.mean();r[f'mondrian_size_{name}']=(((1-pt[g_t])<=qg).astype(int)+(pt[g_t]<=qg)).mean()
            else:r[f'mondrian_cov_{name}']=np.nan;r[f'mondrian_size_{name}']=np.nan
    r['verdict']=verdict(r['coverage'],len(hit),a);return r

def load(f):
    z=np.load(f);return z['mean'],(z['variance'] if 'variance' in z.files else None),z['labels'],z['masks'].astype(bool)

def analyse(selftest=False):
    R,K=[],[]
    for vdir in sorted(glob.glob(str(C/'preds'/'Uni-Mol*'/'*'/'*'))):
        if not (Path(vdir)/'DONE').exists():continue
        bb,split,d=Path(vdir).parts[-3:];reg=d in ('esol','freesolv','lipo','qm7');nv_v,nv_te=novelty(d,split)
        for vf in sorted(glob.glob(vdir+'/*.npz')):
            name=Path(vf).stem;method,seed=name.rsplit('-',1);method=method.replace('_',' ')
            mv,vv,yv,kv=load(vf)
            if selftest:                 # validation halves only; test split untouched
                rng=np.random.default_rng(0);perm=rng.permutation(len(yv));h=len(yv)//2;ci,ti=perm[:h],perm[h:]
                cal=(mv[ci],None if vv is None else vv[ci],yv[ci],kv[ci]);tst=(mv[ti],None if vv is None else vv[ti],yv[ti],kv[ti]);nc,nt=nv_v[ci],nv_v[ti]
            else:
                tf=C/'preds_test'/bb/split/d/Path(vf).name
                if not tf.exists():tf=V/'results'/bb/split/d/Path(vf).name
                if not tf.exists():tf=C.parent/'expanded_v8_gin'/'results'/bb/split/d/Path(vf).name   # v2: GIN reproduction results (path fix; skipped silently before)
                if not tf.exists():tf=C.parent/'expanded_v10_unimol'/'results'/bb/split/d/Path(vf).name   # Uni-Mol reproduction results
                if not tf.exists():continue
                cal=(mv,vv,yv,kv);mt,vt,yt,kt=load(tf);tst=(mt,vt,yt,kt);nc,nt=nv_v,nv_te
            per=[]
            for j in range(cal[2].shape[1]):
                mc,mt_=cal[3][:,j],tst[3][:,j]
                if mc.sum()<MIN_CAL or mt_.sum()==0:continue
                for a in ALPHAS:
                    if reg:
                        for sig in (True,False):
                            r=reg_rows((cal[0][mc,j],cal[1][mc,j],cal[2][mc,j]),(tst[0][mt_,j],tst[1][mt_,j],tst[2][mt_,j]),nc[mc],nt[mt_],a,sig)
                            per.append({'alpha':a,'score':'normalised' if sig else 'unnormalised','task':j,**r})
                    else:
                        if len(np.unique(cal[2][mc,j]))<2:continue
                        for cc in (False,True):
                            r=cls_rows((cal[0][mc,j],cal[2][mc,j]),(tst[0][mt_,j],tst[2][mt_,j]),nc[mc],nt[mt_],a,cc)
                            per.append({'alpha':a,'score':'LAC class-conditional' if cc else 'LAC','task':j,**r})
            if not per:continue
            df=pd.DataFrame(per);num=[c for c in df.columns if c not in ('alpha','score','task','verdict')]
            agg=df.groupby(['alpha','score'])[num].mean(numeric_only=True).reset_index()
            agg['tasks']=df.groupby(['alpha','score']).task.count().values
            agg['verdict_held_tasks']=df.groupby(['alpha','score']).verdict.apply(lambda s:(s=='held').sum()).values
            agg['verdict_under_tasks']=df.groupby(['alpha','score']).verdict.apply(lambda s:(s=='under').sum()).values
            for _,row in agg.iterrows():(R if reg else K).append({'backbone':bb,'split':split,'dataset':d,'method':method,'seed':seed,**row.to_dict()})
    out=C/('selftest_unimol' if selftest else 'results_unimol');out.mkdir(exist_ok=True)
    pd.DataFrame(R).to_csv(out/'conformal_regression.csv',index=False);pd.DataFrame(K).to_csv(out/'conformal_classification.csv',index=False)
    print('rows',len(R),len(K),'->',out)

if __name__=='__main__':
    mode=sys.argv[1]
    analyse(selftest=(mode=='selftest'))
