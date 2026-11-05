"""Item B analysis (rules in PROTOCOL.md): R10 verdicts per backbone and dataset, comparison with the original three-repeat verdicts, family sign-flip test with Holm over the four backbones, sanity check against the earlier per-seed Laplace NLL."""
import json,itertools,numpy as np,pandas as pd
from pathlib import Path
from scipy import stats
D=Path(__file__).resolve().parent;W=D.parent
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'}
EARLIER={'DNN':lambda d:W/'expanded_v6_laplace'/'results'/'scaffold'/d,'ChemBERTa':lambda d:W/'expanded_v6_laplace'/'chemberta'/'results'/'scaffold'/d,'GIN':lambda d:W/'expanded_v8_gin'/'laplace'/'results'/'scaffold'/d,'Uni-Mol':lambda d:W/'expanded_v10_unimol'/'laplace'/'results'/'scaffold'/d}
ORIG={'DNN':[0,100,200],'ChemBERTa':[0,100,200],'GIN':[0,100,200],'Uni-Mol':[0,1,2]}
def verdict(d):
    d=np.asarray(d);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);h=stats.t.ppf(.975,n-1)*se;neg=(d<0).mean();pos=(d>0).mean()
    v='improves' if (neg>=.9 and m+h<0) else ('worsens' if (pos>=.9 and m-h>0) else 'unclear');return v,m,se,m-h,m+h,neg
rows=[];sanity=[];fam=[]
for bb in ['DNN','ChemBERTa','GIN','Uni-Mol']:
    if not all((D/'results'/bb/f'{d}.json').exists() for d in DS):print('skip',bb,'(not complete)');continue
    zs=[]
    for d in DS:
        R=json.loads((D/'results'/bb/f'{d}.json').read_text());diff=np.array([r['laplace']['nll']-r['deterministic']['nll'] for r in R]);seeds=[r['seed'] for r in R]
        v,m,se,lo,hi,neg=verdict(diff);orig=[x for s,x in zip(seeds,diff) if s in ORIG[bb]];vo='improves' if all(x<-1e-9 for x in orig) else ('worsens' if all(x>1e-9 for x in orig) else 'mixed')
        cal='ce' if d in REG else 'ece';dc=np.array([r['laplace'][cal]-r['deterministic'][cal] for r in R])
        rows.append({'backbone':bb,'dataset':d,'n_repeats':len(diff),'mean_dNLL':m,'se':se,'ci_lo':lo,'ci_hi':hi,'frac_negative':neg,'verdict_R10':v,'verdict_original':vo,'agree':(v==vo) or (v=='improves' and vo=='improves'),'mean_d_calibration':dc.mean(),'cal_metric':cal,'edge_tasks_total':sum(r['alpha_edge_tasks'] for r in R),'tasks_total':sum(r['tasks'] for r in R)})
        zs.append(m/se if se>0 else 0.0)
        old=json.loads((EARLIER[bb](d)/'per_seed_metrics.json').read_text())
        for r in R:
            if r['seed'] in ORIG[bb]:
                o=[x for x in old if x['method']=='Laplace' and int(x['seed'])==r['seed'] and x.get('subgroup','all')=='all'][0]['nll'];sanity.append(abs(o-r['laplace']['nll']))
    z=np.array(zs);obs=z.sum();flips=np.array([np.sum(z*np.array(s)) for s in itertools.product([1,-1],repeat=len(z))]);p=float((flips<=obs+1e-12).mean());fam.append({'backbone':bb,'sum_t':round(float(obs),2),'p_one_sided':round(p,4)})
Rd=pd.DataFrame(rows);Rd.to_csv(D/'laplace10_by_dataset.csv',index=False)
F=pd.DataFrame(fam)
if len(F):
    o=np.argsort(F.p_one_sided.values);m=len(F);adj=np.zeros(m);run=0
    for k,i in enumerate(o):run=max(run,min(1,F.p_one_sided.values[i]*(m-k)));adj[i]=run
    F['holm_p']=adj.round(4);F.to_csv(D/'laplace10_family.csv',index=False)
pd.set_option('display.width',250)
print(Rd[['backbone','dataset','n_repeats','mean_dNLL','ci_lo','ci_hi','frac_negative','verdict_R10','verdict_original','mean_d_calibration']].round(4).to_string(index=False))
print(Rd.groupby('backbone').agg(improves=('verdict_R10',lambda s:int((s=='improves').sum())),worsens=('verdict_R10',lambda s:int((s=='worsens').sum())),unclear=('verdict_R10',lambda s:int((s=='unclear').sum())),orig_improves=('verdict_original',lambda s:int((s=='improves').sum())),mean_gain=('mean_dNLL','mean'),edge=('edge_tasks_total','sum'),tasks=('tasks_total','sum')).round(4).to_string())
print('verdict agreement R10 vs original per backbone:',Rd.groupby('backbone').apply(lambda g:int((g.verdict_R10==g.verdict_original.replace({'mixed':'unclear'})).sum())).to_dict())
print(F.to_string(index=False));print('max |recomputed - earlier| Laplace NLL on original seeds:',max(sanity) if sanity else None,'over',len(sanity))
