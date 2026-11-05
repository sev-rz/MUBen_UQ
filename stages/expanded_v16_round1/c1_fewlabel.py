"""C1: few labelled target molecules (PROTOCOL.md). Saved predictions only.  python c1_fewlabel.py [--bb DNN ChemBERTa GIN] [--datasets d ...] [--R 50]"""
import sys,argparse;sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from cp_common import *
ap=argparse.ArgumentParser();ap.add_argument('--bb',nargs='*',default=['DNN','ChemBERTa','GIN']);ap.add_argument('--datasets',nargs='*',default=DS);ap.add_argument('--R',type=int,default=50);a=ap.parse_args()
KS=[0,5,10,20,50,100];rows=[];OUT=Path(__file__).resolve().parent
def pooled_q(sv,sl):return qhat(np.concatenate([sv,sl]))
def balanced_q(sv,sl):return wqhat(np.concatenate([sv,sl]),np.concatenate([np.full(len(sv),0.5/len(sv)),np.full(len(sl),0.5/len(sl))]))
for d in a.datasets:
    reg=d in REG;gr=None
    for bb in a.bb:
        for method,seed in units(bb):
            vf,tf=paths(bb,method,d,seed)
            if not (vf.exists() and tf.exists()):continue
            mt,vt,yt,kt,idt=load(tf);mv,vv,yv,kv,_=load(vf)
            if gr is None:
                gr=groups(d,idt);u=np.unique(gr);gi=np.searchsorted(u,gr);sizes=np.bincount(gi,minlength=len(u))
                rng=np.random.default_rng(int(hashlib.md5(d.encode()).hexdigest()[:8],16));perms=[rng.permutation(len(u)) for _ in range(a.R)]
                masks={}
                for r,p in enumerate(perms):
                    cs=np.cumsum(sizes[p])
                    for k in KS:
                        if k==0:masks[(r,k)]=(np.zeros(len(gr),bool),np.ones(len(gr),bool));continue
                        m=int(np.searchsorted(cs,k,side='left'))+1;chosen=p[:m];L=np.isin(gi,chosen);masks[(r,k)]=(L,~L)
            acc={}
            for (r,k),(Lm,Em) in masks.items():
                for j in range(yt.shape[1]):
                    ok=kt[:,j];E=Em&ok;L=Lm&ok;kvj=kv[:,j]
                    if E.sum()<MINCAL:continue
                    if reg:
                        sv=reg_scores(mv[kvj,j],vv[kvj,j],yv[kvj,j]);sl=reg_scores(mt[L,j],vt[L,j],yt[L,j]) if L.any() else np.array([])
                    else:
                        if len(np.unique(yv[kvj,j]))<2:continue
                        sv=cls_scores(mv[kvj,j],yv[kvj,j]);sl=cls_scores(mt[L,j],yt[L,j]) if L.any() else np.array([])
                    arms={'base':qhat(sv)}
                    if k>0 and len(sl)>0:
                        arms['pooled']=pooled_q(sv,sl)
                        if k>=5:arms['balanced']=balanced_q(sv,sl)
                        if k>=20 and len(sl)>=MINCAL:arms['target']=qhat(sl)
                    for arm,q in arms.items():
                        c,s=(reg_cov(mt[E,j],vt[E,j],yt[E,j],q) if reg else cls_cov(mt[E,j],yt[E,j],q))
                        acc.setdefault((k,arm,j),[]).append((c,s,len(sl)))
            df=pd.DataFrame([{'k':k,'arm':arm,'task':j,'cov':np.mean([x[0] for x in v]),'size':np.nanmean([x[1] for x in v]) if not all(np.isnan(x[1]) for x in v) else np.nan,'n_lab':np.mean([x[2] for x in v]),'reps':len(v)} for (k,arm,j),v in acc.items()])
            t=df.groupby(['k','arm'])[['cov','size','n_lab']].mean().reset_index()
            for _,r in t.iterrows():rows.append({'backbone':bb,'dataset':d,'kind':'reg' if reg else 'cls','method':method,'seed':seed,'k':int(r['k']),'arm':r['arm'],'cov':r['cov'],'size':r['size'],'n_lab':r['n_lab']})
    print('done',d,flush=True)
R=pd.DataFrame(rows);R.to_csv(OUT/'c1_by_unit.csv',index=False)
Dd=R.groupby(['backbone','kind','method','dataset','k','arm'])[['cov','size','n_lab']].mean().reset_index()
S=Dd.groupby(['backbone','kind','method','k','arm'])[['cov','size']].mean().reset_index();S.to_csv(OUT/'c1_summary.csv',index=False)
base=S[S.arm=='base'][['backbone','kind','method','k','cov','size']].rename(columns={'cov':'cov_base','size':'size_base'})
S=S.merge(base,on=['backbone','kind','method','k'],how='left');S['size_ratio']=S['size']/S['size_base'];S.to_csv(OUT/'c1_summary.csv',index=False)
pd.set_option('display.width',220)
print(S[(S.kind=='cls')&(S.arm.isin(['base','pooled','balanced','target']))&(S.method.isin(['Deterministic','Laplace']))].round(3).to_string(index=False))
