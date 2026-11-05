"""C6: conformal under the linear learning-rate schedule (ChemBERTa, scaffold; PROTOCOL.md). Validation predictions come from the saved linear-schedule nets and posteriors (test split not touched);
test predictions are the saved ones of item 9. Then the n0089 recipe (alpha 0.1) for Deterministic and Laplace, linear versus constant schedule."""
import os,sys,json
from pathlib import Path
R1=Path(__file__).resolve().parent;W=R1.parent;V13=W/'expanded_v13_schedule';sys.path.insert(0,str(R1));sys.path.insert(0,str(V13/'laplace'))
import chemberta_laplace_lin as M
import numpy as np,torch
from cp_common import *
cb=M.cb;L=M.L;t=cb.t;ev=L.ev;OUT=V13/'laplace'/'results'/'scaffold';VAL=R1/'c6_valid_lin';VAL.mkdir(exist_ok=True)
for d in DS:
    o=VAL/d;o.mkdir(exist_ok=True)
    if (o/'DONE').exists():continue
    c=cb.CCtx('scaffold',d);tt=t.table(d);v=t.split_index(d,'scaffold')['valid'];y=tt['y'][v].astype(float);m=tt['m'][v];ids=c.ids['valid']
    for sd in [0,100,200]:
        net,_=cb.fit(c,'det',sd);mean,var=ev.convert(c,cb.predict(net,c.tokens['valid']));a=dict(mean=mean,labels=y,masks=m,ids=ids)
        if var is not None:a['variance']=var
        np.savez_compressed(o/f'Deterministic-{sd}.npz',**a)
        with np.load(OUT/d/f'seed-{sd}'/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
        mu,vv,noise=M.part_outputs(c,net,arr,'valid');pm,pv=L.compose(c,mu,vv,noise,torch.zeros(len(y)),None,0.);mn,vr=L._convert(c,pm,pv);a=dict(mean=mn,labels=y,masks=m,ids=ids)
        if vr is not None:a['variance']=vr
        np.savez_compressed(o/f'Laplace-{sd}.npz',**a)
    (o/'DONE').write_text('1');print('valid preds',d,flush=True)
rows=[]
for sched in ['constant','linear']:
    for method in ['Deterministic','Laplace']:
        for d in DS:
            reg=d in REG;res=[]
            for sd in [0,100,200]:
                if sched=='linear':vf=VAL/d/f'{method}-{sd}.npz';tf=OUT/d/f'{method}-{sd}.npz'
                else:vf,tf=paths('ChemBERTa',method,d,sd)
                mt,vt,yt,kt,_=load(tf);mv,vv,yv,kv,_=load(vf);per=[]
                for j in range(yt.shape[1]):
                    ok=kt[:,j];kvj=kv[:,j]
                    if ok.sum()<MINCAL:continue
                    if reg:q=qhat(reg_scores(mv[kvj,j],vv[kvj,j],yv[kvj,j]));per.append(reg_cov(mt[ok,j],vt[ok,j],yt[ok,j],q))
                    else:
                        if len(np.unique(yv[kvj,j]))<2:continue
                        q=qhat(cls_scores(mv[kvj,j],yv[kvj,j]));per.append(cls_cov(mt[ok,j],yt[ok,j],q))
                if per:res.append(np.nanmean(np.array(per),0))
            if res:r=np.nanmean(np.array(res),0);rows.append({'schedule':sched,'method':method,'dataset':d,'kind':'reg' if reg else 'cls','cov':r[0],'size':r[1]})
Rr=pd.DataFrame(rows);Rr.to_csv(R1/'c6_by_dataset.csv',index=False)
S=Rr.groupby(['kind','method','schedule'])[['cov','size']].mean().reset_index();S.to_csv(R1/'c6_summary.csv',index=False)
pd.set_option('display.width',200);print(S.round(3).to_string(index=False))
print(Rr[Rr.kind=='cls'].pivot_table(index=['method','dataset'],columns='schedule',values='cov').round(3).to_string())
