"""Protocol rule 4 cross-device check: BACE scaffold ChemBERTa, v4 GPU (batch 128) vs expanded_v3 CPU (micro-batches of 8), test side.
Run only after ChemBERTa_scaffold_bace FROZEN + evaluated. Reads expanded_v3/chemberta_results read-only; writes results/CROSSDEVICE_BACE.json."""
import json, statistics as st
from pathlib import Path
R=Path(__file__).resolve().parent;V3=R.parent/'expanded_v3'/'chemberta_results'/'per_seed_metrics.json'
assert (R/'frozen'/'ChemBERTa_scaffold_bace.json').exists()
v3=json.loads(V3.read_text());v4=json.loads((R/'results/ChemBERTa/scaffold/bace/per_seed_metrics.json').read_text())
out={'v3_source':str(V3.relative_to(R.parent)),'seeds':[0,100,200],'methods':{}}
for m in ('Deterministic','MC dropout','Temperature'):
    a={r['seed']:r for r in v3 if r['method']==m};b={r['seed']:r for r in v4 if r['method']==m and r['seed'] in (0,100,200)}
    rows=[{'seed':s,'roc_auc_v3_cpu':a[s]['roc_auc'],'roc_auc_v4_gpu':b[s]['roc-auc'],'nll_v3_cpu':a[s]['nll'],'nll_v4_gpu':b[s]['nll'],
           'brier_v3_cpu':a[s]['brier'],'brier_v4_gpu':b[s]['brier']} for s in (0,100,200)]
    x=[r['roc_auc_v3_cpu'] for r in rows];y=[r['roc_auc_v4_gpu'] for r in rows]
    out['methods'][m]={'per_seed':rows,'roc_auc_mean_v3':st.mean(x),'roc_auc_mean_v4':st.mean(y),'roc_auc_mean_diff':st.mean(y)-st.mean(x),
                       'roc_auc_seed_sd_v3':st.stdev(x),'roc_auc_seed_sd_v4':st.stdev(y),'max_abs_seed_diff':max(abs(p-q) for p,q in zip(x,y)),
                       'nll_mean_v3':st.mean(r['nll_v3_cpu'] for r in rows),'nll_mean_v4':st.mean(r['nll_v4_gpu'] for r in rows)}
(R/'results'/'CROSSDEVICE_BACE.json').write_text(json.dumps(out,indent=1))
for m,v in out['methods'].items():print(m,'AUC v3 %.4f v4 %.4f diff %+.4f (seed sd %.3f / %.3f), max |seed diff| %.3f; NLL %.3f -> %.3f'%(v['roc_auc_mean_v3'],v['roc_auc_mean_v4'],v['roc_auc_mean_diff'],v['roc_auc_seed_sd_v3'],v['roc_auc_seed_sd_v4'],v['max_abs_seed_diff'],v['nll_mean_v3'],v['nll_mean_v4']))
