"""Phase-3 comparison tables from saved predictions (no training, no test-driven choices).

random   : recompute random-split metrics from saved npz, verify stored rows, compare to
           authors' random tables (reports/random) in the format of work/reproduction_comparison.csv.
bayesian : BBP/SGLD ESOL rows vs authors' scaffold-split ESOL rows (reports/primary).
chemberta: ChemBERTa BACE vs v2 DNN BACE and authors' ChemBERTa/DNN BACE rows; runs only
           after chemberta_results/per_seed_metrics.json holds all three seeds.
"""
from pathlib import Path
import sys,csv,json,argparse
ROOT=Path(__file__).resolve().parent;V2=ROOT.parent/'expanded_v2'
sys.path.insert(0,str(V2))
import run_expansion as e
import numpy as np

import os;OUT=Path(os.environ.get('PHASE3_OUT',ROOT/'results'));COMMIT='f0ec55667c1c4cc7b63369018fd011a96f31fdc4'
NAME={'Deterministic':'none','Temperature':'TemperatureScaling','MC dropout':'MCDropout','Ensemble 5':'DeepEnsembles','Focal Loss':'FocalLoss','SWAG':'SWAG'}
CLS={'roc_auc':'roc-auc','prc_auc_trapezoid':'prc-auc','ece15_positive':'ece','mce':'mce','nll':'nll','brier':'brier'}
REG={'rmse':'rmse','mae':'mae','nll':'nll','ce_upstream':'ce'}
FIELDS=['dataset','method','metric','upstream_mean','upstream_std','reproduced_mean','reproduced_std','delta_reproduced_minus_upstream','comparison_status','source','source_commit']

def csvwrite(path,items):
    keys=list(dict.fromkeys(k for r in items for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
def refs(kind,backbone,d):
    return {r['method']:r for r in csv.DictReader((ROOT/'references'/kind/f'{backbone}-{d}.csv').open())}
def summarize(rows,keys=('dataset','method')):
    out=[]
    for group in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
        rs=[r for r in rows if tuple(r[k] for k in keys)==group];sm=dict(zip(keys,group));sm['n_repeats']=len(rs)
        for k in rs[0]:
            if k in keys or k=='seed' or not isinstance(rs[0][k],(int,float)):continue
            v=[r[k] for r in rs if k in r];sm[k+'_mean']=float(np.mean(v));sm[k+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else float('nan')
        out.append(sm)
    return out
def verify(folder,rows):
    """Recompute every stored metric from its saved prediction file."""
    for r in rows:
        with np.load(folder/f'{r["dataset"]}-{r["seed"]}-{r["method"].replace(" ","_")}.npz',allow_pickle=False) as z:
            m,_=e.reg_metrics(z['labels'],z['mean'],z['variance'],z['ids']) if r['dataset']=='esol' else e.class_metrics(z['labels'],z['mean'],z['ids'])
        for k,v in m.items():np.testing.assert_allclose(v,r[k],rtol=1e-10,atol=1e-12)
    return len(rows)

def random_splits():
    rows=[];checked=0
    for s in [0,1,2]:
        folder=ROOT/'random'/f'split-{s}';rr=json.loads((folder/'results/per_seed_metrics.json').read_text())
        checked+=verify(folder/'results',rr);rows+=[{**r,'split':s} for r in rr]
    assert len(rows)==93==checked
    audit={s:json.loads((ROOT/'random'/f'split-{s}'/'results/split_audit.json').read_text()) for s in [0,1,2]}
    summary=summarize([{k:v for k,v in r.items() if k!='split'} for r in rows])
    comp=[]
    for d in e.DATASETS:
        ref=refs('random','DNN-rdkit',d);metrics=REG if d=='esol' else CLS
        for sm in summary:
            if sm['dataset']!=d or sm['method'] not in NAME:continue
            r=ref['DNN-rdkit-'+NAME[sm['method']]]
            for m,old in metrics.items():
                if old+'-mean' not in r:continue
                up=float(r[old+'-mean']);cur=sm[m+'_mean']
                comp.append({'dataset':d,'method':sm['method'],'metric':m,'upstream_mean':up,'upstream_std':r.get(old+'-std',''),'reproduced_mean':cur,'reproduced_std':sm[m+'_std'],'delta_reproduced_minus_upstream':cur-up,
                    'comparison_status':'random_split_scoped_not_exact_replication;3_regenerated_splits;SD_across_splits'+(';ensemble5_vs_paper' if sm['method']=='Ensemble 5' else ''),
                    'source':f'reports/random/DNN-rdkit-{d}.csv','source_commit':COMMIT})
    csvwrite(OUT/'random_split_per_split_metrics.csv',rows);csvwrite(OUT/'random_split_summary.csv',summary)
    csvwrite(OUT/'random_split_comparison.csv',[{k:c[k] for k in FIELDS} for c in comp])
    e.save_json(OUT/'random_split_audit.json',audit)
    # Compact view: z = delta / published SD (descriptive only; split sets are not identical).
    for c in comp:
        sd=float(c['upstream_std']) if c['upstream_std'] not in ('',None) else float('nan')
        if c['metric'] in ('roc_auc','rmse','nll','ece15_positive','ce_upstream'):
            print(f"{c['dataset']:5s} {c['method']:13s} {c['metric']:14s} paper {c['upstream_mean']:.3f}±{sd:.3f} ours {c['reproduced_mean']:.3f}±{c['reproduced_std']:.3f} d={c['delta_reproduced_minus_upstream']:+.3f}")
    return comp

def bayesian():
    rows=json.loads((ROOT/'bayesian_results/per_seed_metrics.json').read_text());assert len(rows)==9
    for r in rows:
        with np.load(ROOT/'bayesian_results'/f'{r["seed"]}-{r["method"].replace(" ","_")}.npz',allow_pickle=False) as z:
            m,_=e.reg_metrics(z['labels'],z['mean'],z['variance'],z['ids'])
        for k,v in m.items():np.testing.assert_allclose(v,r[k],rtol=1e-10,atol=1e-12)
    summary=summarize(rows);ref=refs('primary','DNN-rdkit','esol');comp=[]
    for sm in summary:
        r=ref['DNN-rdkit-'+('BBP' if sm['method']=='BBP' else 'SGLD')]
        for m,old,status in [('rmse','rmse','same_definition'),('mae','mae','same_definition'),('nll','nll','same_definition_constant_omitted'),('nll_full','nll','ours_includes_0.5log2pi;paper_omits'),('ce_upstream','ce','same_source_formula_all_pairs'),('ce_paired','ce','different_definition_paired_PIT')]:
            up=float(r[old+'-mean']);cur=sm[m+'_mean']
            comp.append({'dataset':'esol','method':sm['method'],'metric':m,'upstream_mean':up,'upstream_std':r.get(old+'-std',''),'reproduced_mean':cur,'reproduced_std':sm[m+'_std'],'delta_reproduced_minus_upstream':cur-up,
                'comparison_status':'scaffold_split;3_seeds;scoped_not_exact_replication;'+status,'source':'reports/primary/DNN-rdkit-esol.csv','source_commit':COMMIT})
    csvwrite(OUT/'bayesian_summary.csv',summary);csvwrite(OUT/'bayesian_comparison.csv',[{k:c[k] for k in FIELDS} for c in comp])
    for c in comp:print(f"{c['method']:22s} {c['metric']:12s} paper {c['upstream_mean']:.3f} ours {c['reproduced_mean']:.3f}±{c['reproduced_std']:.3f}")
    return comp

def chemberta():
    rows=json.loads((ROOT/'chemberta_results/per_seed_metrics.json').read_text())
    assert sorted({r['seed'] for r in rows})==[0,100,200] and len(rows)==9
    for r in rows:
        with np.load(ROOT/'chemberta_results'/f'{r["seed"]}-{r["method"].replace(" ","_")}.npz',allow_pickle=False) as z:
            # train_chemberta.evaluate scored p.astype(float) but saved float32; recompute on the same float64 cast.
            m,_=e.class_metrics(z['labels'].astype(float),z['mean'].astype(float),z['ids'])
        for k,v in m.items():np.testing.assert_allclose(v,r[k],rtol=1e-10,atol=1e-12)
    dnn=[r for r in json.loads((V2/'results/per_seed_metrics.json').read_text()) if r['dataset']=='bace' and r['method'] in ('Deterministic','Temperature','MC dropout')]
    assert len(dnn)==9
    ours={('ChemBERTa',s['method']):s for s in summarize([{k:v for k,v in r.items() if k!='backbone'} for r in rows])}
    ours.update({('DNN',s['method']):s for s in summarize(dnn)})
    paper={'ChemBERTa':refs('primary','ChemBERTa','bace'),'DNN':refs('primary','DNN-rdkit','bace')};comp=[]
    for method in ('Deterministic','Temperature','MC dropout'):
        for m,old in CLS.items():
            up={b:paper[b][('ChemBERTa' if b=='ChemBERTa' else 'DNN-rdkit')+'-'+NAME[method]] for b in paper}
            for b in ('ChemBERTa','DNN'):
                s=ours[(b,method)];u=float(up[b][old+'-mean'])
                comp.append({'dataset':'bace','backbone':b,'method':method,'metric':m,'upstream_mean':u,'upstream_std':up[b].get(old+'-std',''),'reproduced_mean':s[m+'_mean'],'reproduced_std':s[m+'_std'],
                    'delta_reproduced_minus_upstream':s[m+'_mean']-u,'n_seeds':s['n_repeats'],'comparison_status':'scaffold_split;scoped_not_exact_replication'+(';v2_DNN_paper8GELU' if b=='DNN' else ';full_finetune_pooler_output'),
                    'source':f"reports/primary/{'ChemBERTa' if b=='ChemBERTa' else 'DNN-rdkit'}-bace.csv",'source_commit':COMMIT})
            uc=float(up['ChemBERTa'][old+'-mean'])-float(up['DNN'][old+'-mean']);oc=ours[('ChemBERTa',method)][m+'_mean']-ours[('DNN',method)][m+'_mean']
            comp.append({'dataset':'bace','backbone':'ChemBERTa_minus_DNN','method':method,'metric':m,'upstream_mean':uc,'upstream_std':'','reproduced_mean':oc,'reproduced_std':'','delta_reproduced_minus_upstream':oc-uc,'n_seeds':3,
                'comparison_status':'backbone_contrast;same_test_molecules;unpaired_seeds','source':'reports/primary/ChemBERTa-bace.csv - reports/primary/DNN-rdkit-bace.csv','source_commit':COMMIT})
    csvwrite(OUT/'chemberta_comparison.csv',comp)
    for c in comp:
        if c['metric'] in ('roc_auc','nll','ece15_positive'):print(f"{c['backbone']:19s} {c['method']:13s} {c['metric']:14s} paper {c['upstream_mean']:+.3f} ours {c['reproduced_mean']:+.3f}")
    return comp

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('parts',nargs='+',choices=['random','bayesian','chemberta']);OUT.mkdir(exist_ok=True)
    for p in a.parse_args().parts:print('==',p);{'random':random_splits,'bayesian':bayesian,'chemberta':chemberta}[p]()
