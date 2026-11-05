"""Item 9 analysis (rules in PROTOCOL.md): Q1 Deterministic constant vs linear; Q2 paired Laplace gain under the linear schedule; Q3 gain ratio and gap closing. Exploratory statistics as check E where per-molecule predictions exist."""
import json,itertools,numpy as np,pandas as pd
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;V4=W/'expanded_v4'/'results'/'ChemBERTa'/'scaffold';V6=W/'expanded_v6_laplace'/'chemberta'/'results'/'scaffold'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'};SEEDS=[0,100,200]
def rows(f,method=None):
    r=json.loads(Path(f).read_text());return [x for x in r if (method is None or x['method']==method) and x.get('subgroup','all')=='all']
out=[]
for d in DS:
    cal='ce' if d in REG else 'ece'
    det_c=rows(V4/d/'per_seed_metrics.json','Deterministic');lap_c=rows(V6/d/'per_seed_metrics.json','Laplace')
    det_l=json.loads((D/'metrics'/f'lin_det_{d}.json').read_text());lap_l=rows(D/'laplace'/'results'/'scaffold'/d/'per_seed_metrics.json','Laplace')
    g=lambda R,k:{int(x['seed']):x[k] for x in R};n=lambda R:g(R,'nll')
    dc,lc,dl,ll=n(det_c),n(lap_c),n(det_l),n(lap_l)
    gain_c=[lc[s]-dc[s] for s in SEEDS];gain_l=[ll[s]-dl[s] for s in SEEDS]
    v=lambda x:'improves' if all(a<-1e-9 for a in x) else ('worsens' if all(a>1e-9 for a in x) else 'mixed')
    out.append({'dataset':d,'det_nll_constant':np.mean([dc[s] for s in SEEDS]),'det_nll_linear':np.mean([dl[s] for s in SEEDS]),'lap_nll_constant':np.mean([lc[s] for s in SEEDS]),'lap_nll_linear':np.mean([ll[s] for s in SEEDS]),
                'gain_constant':np.mean(gain_c),'gain_linear':np.mean(gain_l),'verdict_constant':v(gain_c),'verdict_linear':v(gain_l),'gain_ratio_lin_over_const':np.mean(gain_l)/np.mean(gain_c) if abs(np.mean(gain_c))>1e-9 else np.nan,
                'cal_metric':cal,'det_cal_constant':np.mean([g(det_c,cal)[s] for s in SEEDS]),'det_cal_linear':np.mean([g(det_l,cal)[s] for s in SEEDS]),'lap_cal_linear':np.mean([g(lap_l,cal)[s] for s in SEEDS]),
                'laplace_constant_closes_gap':bool(np.mean([lc[s] for s in SEEDS])<=np.mean([dl[s] for s in SEEDS]))})
R=pd.DataFrame(out);R.to_csv(D/'sched_results.csv',index=False);pd.set_option('display.width',250)
print(R[['dataset','det_nll_constant','det_nll_linear','lap_nll_constant','lap_nll_linear','gain_constant','gain_linear','verdict_linear','gain_ratio_lin_over_const','laplace_constant_closes_gap']].round(4).to_string(index=False))
print('Q2: datasets improving under linear schedule:',int((R.verdict_linear=='improves').sum()),'of 8 (constant:',int((R.verdict_constant=='improves').sum()),'of 8)')
rm=R.gain_linear.mean()/R.gain_constant.mean();print('Q3: mean gain constant',round(R.gain_constant.mean(),4),'linear',round(R.gain_linear.mean(),4),'ratio of means',round(rm,3),'->','shrinks' if rm<0.5 else ('similar' if rm<=1.5 else 'grows'))
print('Q1: datasets where linear Deterministic NLL is lower:',int((R.det_nll_linear<R.det_nll_constant).sum()),'of 8; mean change',round((R.det_nll_linear-R.det_nll_constant).mean(),4))
print('Laplace(constant) at or below Deterministic(linear) NLL on',int(R.laplace_constant_closes_gap.sum()),'of 8')
