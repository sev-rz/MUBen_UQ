"""Comparison tables against the authors' CSVs (references/, commit f0ec556) and agreement counts.

Reproduced mean/SD over the 3 seeds (scaffold) or 3 splits (random); SD is population SD as in the source's
aggregate_seeded_results. Scaffold ensembles are one pooled 10-member ensemble (no SD), as published.
Agreement as in expanded_v3: |delta| <= k x published SD, counted over rows with published SD > 0.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
COMMIT='f0ec55667c1c4cc7b63369018fd011a96f31fdc4'
METHODS={'none':'Deterministic','TemperatureScaling':'Temperature','FocalLoss':'Focal Loss','MCDropout':'MC dropout','SWAG':'SWAG','BBP':'BBP','SGLD':'SGLD','DeepEnsembles':'Ensembles'}
PREFIX={'DNN':'DNN-rdkit','ChemBERTa':'ChemBERTa'}

def reproduced():
    rows=[]
    for f in sorted((ROOT/'results').glob('*/*/*/per_seed_metrics.json')):rows+=json.loads(f.read_text())
    return pd.DataFrame(rows)

def compare(split_kind):
    r=reproduced();out=[]
    if r.empty:return pd.DataFrame()
    r=r[r.split.eq('scaffold')] if split_kind=='scaffold' else r[r.split.str.startswith('random')]
    for (bb,d),g in r.groupby(['backbone','dataset']):
        ref_path=ROOT/'references'/('primary' if split_kind=='scaffold' else 'random')/f'{PREFIX[bb]}-{d}.csv'
        ref=pd.read_csv(ref_path);ref['method']=ref.method.str.replace(PREFIX[bb]+'-','',regex=False).map(METHODS)
        for _,u in ref.iterrows():
            mine=g[g.method.eq(u.method)]
            for col in [c[:-5] for c in ref.columns if c.endswith('-mean')]:
                if mine.empty or col not in mine:continue
                v=mine[col].astype(float).to_numpy();sd=u.get(f'{col}-std',np.nan);sd=float(sd) if pd.notna(sd) else np.nan
                rm=float(v.mean());rs=float(v.std()) if len(v)>1 else np.nan;delta=rm-float(u[f'{col}-mean'])
                ok=lambda k:bool(abs(delta)<=k*sd) if sd==sd and sd>0 else None
                out.append({'backbone':bb,'split':split_kind,'dataset':d,'method':u.method,'metric':col,'upstream_mean':float(u[f'{col}-mean']),'upstream_std':sd,
                            'reproduced_mean':rm,'reproduced_std':rs,'n_reproduced':len(v),'delta_reproduced_minus_upstream':delta,'within_1sd':ok(1),'within_2sd':ok(2),
                            'source':str(ref_path.relative_to(ROOT)),'source_commit':COMMIT})
    return pd.DataFrame(out)

def fence(text):return '```\n'+text+'\n```'

def counts(c):
    k=c.dropna(subset=['within_1sd'])
    return k.groupby(['backbone','split','dataset']).agg(n=('within_1sd','size'),within_1sd=('within_1sd','sum'),within_2sd=('within_2sd','sum')).reset_index()

def claims(c):
    """Paper claims, counted per dataset on published and reproduced means (lower is better except ROC-AUC)."""
    out=[]
    def val(g,m,metric,col):
        x=g[(g.method==m)&(g.metric==metric)][col];return float(x.iloc[0]) if len(x) else np.nan
    for (bb,sp),g0 in c.groupby(['backbone','split']):
        for col,who in [('upstream_mean','published'),('reproduced_mean','reproduced')]:
            tally={}
            for d,g in g0.groupby('dataset'):
                reg=d in ['esol','freesolv','lipo','qm7'];prim='rmse' if reg else 'roc-auc';sign=1 if reg else -1
                ms=g[g.metric==prim].set_index('method')[col]
                tally.setdefault('C1 Ensembles best on '+('RMSE/ROC-AUC'),[]).append(bool((sign*ms).idxmin()=='Ensembles'))
                for m in ['MC dropout']+([] if reg else ['Temperature']):
                    tally.setdefault(f'C2 {m} NLL < Deterministic',[]).append(bool(val(g,m,'nll',col)<val(g,'Deterministic','nll',col)))
                if not reg:
                    for m in ['BBP','SGLD']:tally.setdefault(f'C3 {m} ROC-AUC < Deterministic (classification)',[]).append(bool(val(g,m,'roc-auc',col)<val(g,'Deterministic','roc-auc',col)))
            for k,v in tally.items():out.append({'backbone':bb,'split':sp,'claim':k,'source':who,'holds':sum(v),'datasets':len(v)})
    t=pd.DataFrame(out)
    return t.pivot_table(index=['backbone','split','claim'],columns='source',values=['holds','datasets'],aggfunc='first') if len(t) else t

def by_metric(c):
    """Agreement by metric: n rows with published SD > 0, within 1 / 2 SD, median (reproduced - published) / published SD."""
    k=c.dropna(subset=['within_1sd']).copy();k['z']=(k.reproduced_mean-k.upstream_mean)/k.upstream_std
    k['kind']=np.where(k.dataset.isin(['esol','freesolv','lipo','qm7']),'regression','classification')
    return k.groupby(['kind','metric','backbone','split']).agg(n=('z','size'),within_1sd=('within_1sd','sum'),within_2sd=('within_2sd','sum'),median_z=('z','median')).round(2).reset_index()

def crossdevice():
    f=ROOT/'results'/'CROSSDEVICE_BACE.json'
    if not f.exists():return '  Cross-device check (BACE, GPU vs expanded_v3 CPU): not available.'
    j=json.loads(f.read_text())['methods']
    return '  Cross-device check (BACE scaffold, test ROC-AUC mean of seeds 0/100/200, v3 CPU -> v4 GPU): '+'; '.join(f"{m} {v['roc_auc_mean_v3']:.4f} -> {v['roc_auc_mean_v4']:.4f}" for m,v in j.items())+'; seed SD about 0.04, so the shift is about one third of a seed SD (journal n0037).'

if __name__=='__main__':
    (ROOT/'results').mkdir(exist_ok=True)
    for s in ['scaffold','random']:
        c=compare(s);c.to_csv(ROOT/'results'/f'comparison_{s}.csv',index=False)
        if not c.empty:print(counts(c).to_string(index=False))
    parts=[pd.read_csv(ROOT/'results'/f'comparison_{s}.csv') for s in ['scaffold','random'] if (ROOT/'results'/f'comparison_{s}.csv').stat().st_size>1]
    if parts:
        c=pd.concat(parts);n=counts(c);tot=n.groupby(['backbone','split'])[['n','within_1sd','within_2sd']].sum().reset_index()
        cl=claims(c)
        md=['# Tier-B summary (generated by tierb_compare.py; do not edit by hand)','',
            'Agreement: |reproduced mean - published mean| <= k x published SD, over rows with published SD > 0 (scaffold ensembles have no SD).','',
            '## Totals per backbone x split','',fence(tot.to_string(index=False)),'','## Per backbone x split x dataset','',fence(n.to_string(index=False)),'',
            '## Claims check (number of datasets where the claim holds)','',fence(cl.to_string()) if len(cl) else '(none)','',
            '## Agreement by metric (rows with published SD > 0; median z = (reproduced - published) / published SD)','',fence(by_metric(c).to_string(index=False)),'',
            'Reading guide: the published SDs come from 3 seeds/splits and are often small (ChemBERTa scaffold ROC-AUC SD 0.001-0.010; DNN up to 0.048), so a difference of 0.01-0.02 AUC or a few percent RMSE can be several SDs. Deterministic ROC-AUC differs from the published value by a median of 0.007 (max 0.054, DNN ClinTox) and RMSE by a median of 4.6 % (max 11 %) over the 16 cells each (journal n0048). Counts of "within k SD" therefore measure agreement relative to the published seed noise, not closeness in natural units.','',
            '## Deviations and limits','',
            '- DNN: RDKit 2024.09 cannot parse 8 Tox21 SMILES (hypervalent Al); rows kept with zero descriptors (DECISIONS D4).',
            '- DNN architecture is fixed by the protocol (8 x 128 GELU, D3); the source default differs. The QM7 DNN gap (scaffold RMSE 127 vs 143 published, 0/20 rows within 1 SD) is not explained; a sensitivity run would be descriptive only and was not made.',
            '- ChemBERTa runs on one GPU (FP32, TF32 off, deterministic algorithms where available), batch 128 processed whole; expanded_v3 used CPU micro-batches of 8 (D5, D6).',
            crossdevice(),
            '- ChemBERTa units that hit CUDA out-of-memory were re-queued or retried from their seed by an operational wrapper (D6); tierb_chemberta.py is unchanged, so all units share one code digest.',
            '- The first ChemBERTa pilot ran on CPU by mistake (CPU-only torch in work/.venv) and was discarded; the projection in PILOT_TIMING.json is from production GPU units.',
            '- Scaffold ensembles are one pooled 10-member ensemble without published SD, so SD-based agreement counts exclude the Ensembles rows (42 of 48 classification and 20 of 24 regression rows per dataset on scaffold).',
            '- Regression NLL and calibration error (CE) are the metrics that deviate most (see the by-metric table: negative median z means the reproduced value is lower than published). The metric definitions were compared with the pinned source by reading (Gaussian NLL = F.gaussian_nll_loss on observed rows, mean and variance inverse-transformed with the training-label scaler, variance = softplus + 1e-6) and match tierb_dnn/tierb_eval; no cause was found for the gap, so it is reported as an unexplained empirical difference, not as a verified reproduction failure or success.',
            '- ChemBERTa scaffold: ROC-AUC and PRC-AUC are lower than published relative to the small published SDs (median z about -2 and -1.4) and RMSE/MAE higher (median z about +1.7); on the random splits the AUCs agree (median z about 0) and RMSE/MAE are higher by about 0.5 to 1 SD. Cause not investigated.','',
            'Details: DECISIONS.md (D1-D6) and OPERATIONS_LOG.md.']
        (ROOT/'results'/'TIERB_SUMMARY.md').write_text('\n'.join(md)+'\n');print(tot.to_string(index=False))
