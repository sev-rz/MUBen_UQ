"""Analysis of C3 (PROTOCOL.md): evidence-based prior versus validation-tuned prior, Deterministic and recalibration; R10 verdicts and family tests.  python analyze_evidence.py"""
import sys,json;from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from r2_common import verdict,signflip,DS,D
rows=[];alpha=[]
for bb in ['DNN','ChemBERTa','GIN','Uni-Mol']:
    for d in DS:
        f=D/'results_evidence'/bb/f'{d}.json'
        if not f.exists():continue
        r=json.loads(f.read_text());g=lambda k:np.array([x[k]['nll'] for x in r]);L,E,S,Dt=g('laplace'),g('evidence'),g('scaling'),g('deterministic')
        vED=verdict(E-Dt,'improves','worsens');vEP=verdict(E-L,'evidence better','evidence worse');vLD=verdict(L-Dt,'improves','worsens')
        dl=np.concatenate([np.log10(x['alpha_evidence'])-np.log10(x['alpha_primary']) for x in r])
        rows.append({'backbone':bb,'dataset':d,'n':len(r),'mean_LD':vLD[1],'mean_ED':vED[1],'mean_EP':vEP[1],'verdict_LD':vLD[0],'verdict_ED':vED[0],'verdict_EP':vEP[0],'t_ED':vED[5],'median_dlog10alpha':float(np.median(dl)),'frac_alpha_ev_larger':float((dl>0).mean()),
                     'edge_primary_share':sum(x['edge_primary'] for x in r)/sum(x['tasks'] for x in r),'edge_evidence_share':sum(x['edge_evidence'] for x in r)/sum(x['tasks'] for x in r),'max_repro_diff':max((x['repro_diff_vs_itemB'] or 0) for x in r)})
A=pd.DataFrame(rows);A.to_csv(D/'evidence_by_dataset.csv',index=False)
F=[]
for bb,g in A.groupby('backbone'):
    if len(g)<8:continue
    F.append({'backbone':bb,'datasets':len(g),'LD_improves':int((g.verdict_LD=='improves').sum()),'ED_improves':int((g.verdict_ED=='improves').sum()),'ED_worsens':int((g.verdict_ED=='worsens').sum()),'evidence_worse_than_primary':int((g.verdict_EP=='evidence worse').sum()),
              'mean_LD':g.mean_LD.mean(),'mean_ED':g.mean_ED.mean(),'mean_EP':g.mean_EP.mean(),'median_dlog10alpha':g.median_dlog10alpha.median(),'frac_alpha_ev_larger':g.frac_alpha_ev_larger.mean(),'edge_primary_share':g.edge_primary_share.mean(),'edge_evidence_share':g.edge_evidence_share.mean(),'p_signflip_ED':signflip(g.t_ED.values),'max_repro_diff':g.max_repro_diff.max()})
F=pd.DataFrame(F);F.to_csv(D/'evidence_family.csv',index=False)
pd.set_option('display.width',260);pd.set_option('display.max_columns',40)
print(A[['backbone','dataset','mean_LD','mean_ED','mean_EP','verdict_LD','verdict_ED','verdict_EP','median_dlog10alpha','edge_primary_share','edge_evidence_share']].round(3).to_string(index=False));print();print(F.round(4).to_string(index=False))
