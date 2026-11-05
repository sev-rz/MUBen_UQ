"""C2: similarity-weighted conformal prediction under covariate shift (PROTOCOL.md + ADDENDUM). Saved predictions only.  python c2_weighted.py [--bb ...] [--datasets ...]"""
import sys,argparse;sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from cp_common import *
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ap=argparse.ArgumentParser();ap.add_argument('--bb',nargs='*',default=['DNN','ChemBERTa','GIN']);ap.add_argument('--datasets',nargs='*',default=DS);a=ap.parse_args()
OUT=Path(__file__).resolve().parent;GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
def fps(sm):
    X=np.zeros((len(sm),1024),np.float32)
    for i,s in enumerate(sm):
        m=Chem.MolFromSmiles(s)
        if m is not None:X[i]=np.asarray(GEN.GetFingerprintAsNumPy(m),np.float32)
    return X
def odds(Xv,Xt,C):
    X=np.vstack([Xv,Xt]);y=np.r_[np.zeros(len(Xv)),np.ones(len(Xt))];p=np.zeros(len(y))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        lr=LogisticRegression(C=C,class_weight='balanced',max_iter=2000).fit(X[tr],y[tr]);p[te]=lr.predict_proba(X[te])[:,1]
    p=np.clip(p,0.05,0.95);w=p/(1-p);return w[:len(Xv)],w[len(Xv):]
def wcov(sv,wv,wt):
    """per-test-molecule weighted quantile: smallest s with sum_{s_i<=s} w_i >= (1-alpha)(S + w_j); inf if none."""
    o=np.argsort(sv);ss=sv[o];cw=np.cumsum(wv[o]);S=cw[-1];tgt=(1-ALPHA)*(S+wt);i=np.searchsorted(cw,tgt,side='left');return np.where(i<len(ss),ss[np.minimum(i,len(ss)-1)],np.inf)
rows=[];ess=[]
for d in a.datasets:
    reg=d in REG;W_={}
    ref=None
    for bb in a.bb:
        vf,tf=paths(bb,'Deterministic',d,0)
        if vf.exists() and tf.exists():ref=(vf,tf);break
    if ref is None:continue
    _,_,_,_,idv=load(ref[0]);_,_,_,_,idt=load(ref[1]);Xv=fps(smiles_of(d,idv));Xt=fps(smiles_of(d,idt))
    for C in (1.0,0.1):
        wv,wt=odds(Xv,Xt,C);W_[C]=(wv,wt);ess.append({'dataset':d,'C':C,'n_val':len(wv),'kish_ess':float(wv.sum()**2/(wv**2).sum()),'mean_w_val':float(wv.mean()),'mean_w_test':float(wt.mean())})
    for bb in a.bb:
        for method,seed in units(bb):
            vf,tf=paths(bb,method,d,seed)
            if not (vf.exists() and tf.exists()):continue
            mt,vt,yt,kt,idt2=load(tf);mv,vv,yv,kv,idv2=load(vf);assert (idt2==idt).all() and (idv2==idv).all()
            for C,(wv,wt) in W_.items():
                res=[]
                for j in range(yt.shape[1]):
                    ok=kt[:,j];kvj=kv[:,j]
                    if ok.sum()<MINCAL:continue
                    if reg:
                        sv=reg_scores(mv[kvj,j],vv[kvj,j],yv[kvj,j])
                    else:
                        if len(np.unique(yv[kvj,j]))<2:continue
                        sv=cls_scores(mv[kvj,j],yv[kvj,j])
                    q0=qhat(sv);qw=wcov(sv,wv[kvj],wt[ok])
                    if reg:c0,s0=reg_cov(mt[ok,j],vt[ok,j],yt[ok,j],q0);cw_,sw=reg_cov(mt[ok,j],vt[ok,j],yt[ok,j],qw)
                    else:c0,s0=cls_cov(mt[ok,j],yt[ok,j],q0);cw_,sw=cls_cov(mt[ok,j],yt[ok,j],qw)
                    res.append((c0,s0,cw_,sw,float(np.isinf(qw).mean())))
                if res:
                    r=np.nanmean(np.array(res),0);rows.append({'backbone':bb,'dataset':d,'kind':'reg' if reg else 'cls','method':method,'seed':seed,'C':C,'cov_unweighted':r[0],'size_unweighted':r[1],'cov_weighted':r[2],'size_weighted':r[3],'frac_inf_q':r[4]})
    print('done',d,flush=True)
R=pd.DataFrame(rows);R.to_csv(OUT/'c2_by_unit.csv',index=False);pd.DataFrame(ess).to_csv(OUT/'c2_ess.csv',index=False)
Dd=R.groupby(['backbone','kind','method','dataset','C'])[['cov_unweighted','cov_weighted','size_unweighted','size_weighted','frac_inf_q']].mean().reset_index()
S=Dd.groupby(['backbone','kind','method','C'])[['cov_unweighted','cov_weighted','size_unweighted','size_weighted','frac_inf_q']].mean().reset_index();S['size_ratio']=S.size_weighted/S.size_unweighted;S.to_csv(OUT/'c2_summary.csv',index=False)
pd.set_option('display.width',220);print(S.round(3).to_string(index=False));print(pd.DataFrame(ess).round(2).to_string(index=False))
