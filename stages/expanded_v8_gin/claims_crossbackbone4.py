"""Cross-backbone claims with Uni-Mol as a fourth backbone (definitions registered in the journal note 'Cross-backbone claims with Uni-Mol as a fourth backbone', before this computation).
(a4) supported on a dataset if Uni-Mol has the largest Deterministic calibration error of the four backbones; overall if >= 6 of 8.
(b4) gap = Deterministic calibration error - smallest mean calibration error over the published methods of that backbone (Laplace family excluded); supported if the Uni-Mol gap exceeds DNN, GIN and ChemBERTa; overall if >= 6 of 8.
Scaffold only. Calibration error = ECE (classification) / CE (regression); robustness view NLL. Descriptive: Spearman rank correlation of parameter count with Deterministic calibration error per dataset (n = 4)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
V8=Path(__file__).resolve().parent;V4=V8.parent/'expanded_v4';V10=V8.parent/'expanded_v10_unimol'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'}
ROOTS={'DNN':V4/'results'/'DNN'/'scaffold','ChemBERTa':V4/'results'/'ChemBERTa'/'scaffold','GIN':V8/'results'/'GIN'/'scaffold','Uni-Mol':V10/'results'/'Uni-Mol'/'scaffold'}
PARAMS={'DNN':0.158,'GIN':0.26,'ChemBERTa':3.43,'Uni-Mol':47.59}
EXCL={'Laplace','Laplace constant','Novelty only','Novelty Laplace'}
def load(bb):
    rows=[]
    for d in DS:
        for r in json.loads((ROOTS[bb]/d/'per_seed_metrics.json').read_text()):
            if r.get('subgroup','all')!='all' or r['method'] in EXCL:continue
            r['dataset']=d;rows.append(r)
    return pd.DataFrame(rows)
F={b:load(b) for b in ROOTS}
def table(mc,mr):
    out=[]
    for d in DS:
        col=mr if d in REG else mc;rec={'dataset':d,'metric':col};det={};gap={}
        for bb,df in F.items():
            g=df[df.dataset==d];pm=g.groupby('method')[col].mean();dt=g[g.method=='Deterministic'].groupby('seed')[col].mean();assert len(dt)==3,(bb,d,len(dt))
            det[bb]=float(dt.mean());gap[bb]=det[bb]-float(pm.min());rec[f'det_{bb}']=det[bb];rec[f'gap_{bb}']=gap[bb];rec[f'best_method_{bb}']=pm.idxmin()
        rec['a4_supported']=bool(max(det,key=det.get)=='Uni-Mol');rec['b4_supported']=bool(all(gap['Uni-Mol']>gap[b] for b in gap if b!='Uni-Mol'))
        bbs=list(PARAMS);rho=stats.spearmanr([PARAMS[b] for b in bbs],[det[b] for b in bbs]).correlation;rec['spearman_params_vs_det']=float(rho)
        out.append(rec)
    return pd.DataFrame(out)
for name,(mc,mr) in {'calibration':('ece','ce'),'nll':('nll','nll')}.items():
    t=table(mc,mr);t.to_csv(V8/'results'/f'CLAIMS4_{name}.csv',index=False)
    print(name,'| a4 supported',int(t.a4_supported.sum()),'of 8 | b4 supported',int(t.b4_supported.sum()),'of 8 | median Spearman',round(float(t.spearman_params_vs_det.median()),2))
    print(t[['dataset','det_DNN','det_GIN','det_ChemBERTa','det_Uni-Mol','gap_ChemBERTa','gap_Uni-Mol','best_method_Uni-Mol','a4_supported','b4_supported']].round(4).to_string(index=False))
