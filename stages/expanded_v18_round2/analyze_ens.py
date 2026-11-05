"""Analysis of C4 (PROTOCOL.md): Laplace on every ensemble member; per-backbone counts, sign-flip with Holm, comparison with the single-network gain of item B.  python analyze_ens.py"""
import sys,json;from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from r2_common import signflip,DS,D
single=pd.read_csv(D.parent/'expanded_v15_scale_widegrid'/'wide_scale_by_dataset.csv');rows=[]
for bb in ['DNN','ChemBERTa','GIN','Uni-Mol']:
    for d in DS:
        f=D/'results_ens'/bb/f'{d}.json'
        if not f.exists():continue
        r=json.loads(f.read_text());x=r['diffs']
        rows.append({'backbone':bb,'dataset':d,'est_LD':x['LD']['est'],'lo_LD':x['LD']['lo'],'hi_LD':x['LD']['hi'],'z_LD':x['LD']['z'],'verdict_LD':x['LD']['verdict'],'est_LS':x['LS']['est'],'verdict_LS':x['LS']['verdict'],'z_LS':x['LS']['z'],'est_SD':x['SD']['est'],'verdict_SD':x['SD']['verdict'],
                     'single_mean_LD':float(single[(single.backbone==bb)&(single.dataset==d)].mean_LD.iloc[0]),'check_E_D_vs_stored':r['check_E_D_vs_stored']})
A=pd.DataFrame(rows);A.to_csv(D/'ens_by_dataset.csv',index=False)
F=[]
for bb,g in A.groupby('backbone'):
    if len(g)<8:continue
    F.append({'backbone':bb,'datasets':len(g),'LD_improves':int((g.verdict_LD=='improves').sum()),'LD_worsens':int((g.verdict_LD=='worsens').sum()),'LS_improves':int((g.verdict_LS=='improves').sum()),'LS_worsens':int((g.verdict_LS=='worsens').sum()),'SD_improves':int((g.verdict_SD=='improves').sum()),
              'mean_ens_LD':g.est_LD.mean(),'mean_single_LD':g.single_mean_LD.mean(),'ens_gain_smaller':bool(abs(g.est_LD.mean())<abs(g.single_mean_LD.mean())),'mean_ens_LS':g.est_LS.mean(),'mean_ens_SD':g.est_SD.mean(),'p_signflip_LD':signflip(g.z_LD.values),'p_signflip_LS_laplace_better':signflip(g.z_LS.values),'max_check':g.check_E_D_vs_stored.max()})
F=pd.DataFrame(F).sort_values('p_signflip_LD').reset_index(drop=True);m=len(F)
F['holm_p_LD']=[round(min(1.0,max(F.p_signflip_LD[:i+1]*np.array([m-k for k in range(i+1)]))),4) for i in range(m)];F.to_csv(D/'ens_family.csv',index=False)
pd.set_option('display.width',260);pd.set_option('display.max_columns',40)
print(A.round(3).to_string(index=False));print();print(F.round(4).to_string(index=False))
