"""Check F (n0096): shift-aware (test-internal) conformal calibration. Test molecules are grouped by Bemis-Murcko scaffold; scaffold groups are assigned at random 50/50 to a calibration half and an evaluation half
(50 assignments, both directions). Split conformal (alpha 0.1; LAC for classification, normalised score for regression) is calibrated on one half and evaluated on the other, and compared with the validation-calibrated sets on
the SAME evaluation halves. Methods: Deterministic, Laplace, Ensembles; three backbones; scaffold and random-0..2. Saved predictions only."""
import csv, math, hashlib, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger; from rdkit.Chem.Scaffolds import MurckoScaffold; RDLogger.DisableLog('rdApp.*')
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');V4=W/'expanded_v4';V5=W/'expanded_v5_conformal';V8=W/'expanded_v8_gin';V10=W/'expanded_v10_unimol'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'};ALPHA=0.1;R=50;MINCAL=20
SPLITS={'scaffold':('scaffold',[0,1,2])}
def qhat(s,a=ALPHA):
    s=np.sort(s);n=len(s);k=math.ceil((n+1)*(1-a));return np.inf if k>n else s[k-1]
def load(f):
    z=np.load(f,allow_pickle=True);return z['mean'].astype(float),(z['variance'].astype(float) if 'variance' in z.files else None),z['labels'].astype(float),z['masks'].astype(bool),z['ids']
def paths(bb,method,sp,d,seed):
    nm=('Ensembles-pooled' if sp=='scaffold' else f'Ensembles-{seed}') if method=='Ensembles' else f'{method}-{seed}'
    if method=='Laplace':val=V5/'preds'/f'{bb}-Laplace'/sp/d/f'{nm}.npz';tst=V5/'preds_test'/f'{bb}-Laplace'/sp/d/f'{nm}.npz'
    else:
        val=V5/'preds'/bb/sp/d/f'{nm}.npz';tst=V10/'results'/'Uni-Mol'/sp/d/f'{nm}.npz'
    return val,tst
_smiles={}
def smiles_of(d,ids):
    if d not in _smiles:_smiles[d]={p:[r['smiles'] for r in csv.DictReader((V4/'data'/d/f'{p}.csv').open())] for p in ['train','valid','test']}
    out=[]
    for i in ids:
        _,part,row=str(i).split('/');out.append(_smiles[d][part][int(row)])
    return out
def groups(d,ids):
    g=[];
    for k,s in enumerate(smiles_of(d,ids)):
        m=Chem.MolFromSmiles(s);g.append(f'unparsable{k}' if m is None else MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    return np.array(g)
def halves(gr,d,sp):
    seed=int(hashlib.md5(f'{d}-{sp}'.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed);u=np.unique(gr);out=[]
    for _ in range(R):
        for _try in range(20):
            a=set(u[rng.random(len(u))<0.5]);inA=np.array([x in a for x in gr])
            if inA.sum()>=MINCAL and (~inA).sum()>=MINCAL:out.append(inA);break
    return out
def reg_run(cal,ev,val):
    (mc,vc,yc),(mt,vt,yt),(mv,vv,yv)=cal,ev,val
    q=qhat(np.abs(yc-mc)/np.sqrt(vc));qv=qhat(np.abs(yv-mv)/np.sqrt(vv));sd=np.sqrt(vt);e=np.abs(yt-mt)
    return {'cov_t':float((e<=q*sd).mean()),'cov_v':float((e<=qv*sd).mean()),'size_t':float((2*q*sd).mean()),'size_v':float((2*qv*sd).mean())}
def cls_run(pc,yc,pt,yt,pv,yv):
    if len(yc)<MINCAL or len(np.unique(yc))<2 or len(yt)==0:return None
    sc=np.where(yc==1,1-pc,pc);sv=np.where(yv==1,1-pv,pv);q=qhat(sc);qv=qhat(sv);o={}
    for tag,qq in [('t',q),('v',qv)]:
        in1=(1-pt)<=qq;in0=pt<=qq;o['cov_'+tag]=float(np.where(yt==1,in1,in0).mean());o['size_'+tag]=float((in1.astype(int)+in0).mean())
    return o
rows=[]
for bb in ['Uni-Mol']:
    for d in DS:
        reg=d in REG
        for sp,(st,seeds) in SPLITS.items():
            ids=None;gr=None;hv=None
            for method in ['Deterministic','Laplace','Ensembles']:
                for seed in (seeds if method!='Ensembles' or sp!='scaffold' else ['pooled']):
                    s_=seed if method!='Ensembles' else (seeds[0] if sp!='scaffold' else 'pooled')
                    if method=='Ensembles' and sp!='scaffold':s_=seeds[0]
                    vf,tf=paths(bb,method,sp,d,s_)
                    if not (vf.exists() and tf.exists()):continue
                    mt,vt,yt,kt,idt=load(tf);mv,vv,yv,kv,_=load(vf)
                    if gr is None:ids=idt;gr=groups(d,ids);hv=halves(gr,d,sp)
                    res=[]
                    for inA in hv:
                        for cal_m in (inA,~inA):
                            ev_m=~cal_m
                            for j in range(yt.shape[1]):
                                c=cal_m&kt[:,j];e=ev_m&kt[:,j];kvj=kv[:,j]
                                if reg:
                                    if c.sum()<MINCAL or e.sum()==0:continue
                                    r=reg_run((mt[c,j],vt[c,j],yt[c,j]),(mt[e,j],vt[e,j],yt[e,j]),(mv[kvj,j],vv[kvj,j],yv[kvj,j]))
                                else:
                                    if len(np.unique(yv[kvj,j]))<2:continue
                                    r=cls_run(mt[c,j],yt[c,j],mt[e,j],yt[e,j],mv[kvj,j],yv[kvj,j])
                                if r:r['task']=j;res.append(r)
                    if not res:continue
                    df=pd.DataFrame(res);t=df.groupby('task')[['cov_t','cov_v','size_t','size_v']].mean().mean()
                    rows.append({'backbone':bb,'dataset':d,'split':sp,'stype':st,'kind':'reg' if reg else 'cls','method':method,'seed':s_,**t.to_dict(),'runs':len(df)})
Rr=pd.DataFrame(rows);Rr.to_csv(W/'expanded_v9_checks'/'conformal_F_unimol_groups.csv',index=False)
D=Rr.groupby(['backbone','stype','kind','method','dataset'])[['cov_t','cov_v','size_t','size_v']].mean().reset_index()
S=D.groupby(['backbone','stype','kind','method'])[['cov_t','cov_v','size_t','size_v']].mean().reset_index();S['size_ratio_t_over_v']=S.size_t/S.size_v
S.to_csv(W/'expanded_v9_checks'/'conformal_F_unimol_summary.csv',index=False);D.to_csv(W/'expanded_v9_checks'/'conformal_F_unimol_by_dataset.csv',index=False)
pd.set_option('display.width',200);print('groups',len(Rr),'| rows in summary',len(S));print(S.round(3).to_string(index=False))
