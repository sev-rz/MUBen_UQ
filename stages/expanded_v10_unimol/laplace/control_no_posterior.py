"""Control for the conformer-averaging deviation (laplace/PROTOCOL.md section 2): posterior-free prediction from the conformer-averaged features (Laplace with zero covariance) versus the Deterministic comparator (authors' averaging of transformed outputs) and versus Laplace. Test NLL per seed; writes control_no_posterior.json. Registered as a control after the Uni-Mol Laplace results were read (n0124); it changes no result."""
import os,sys,json
from pathlib import Path
CB=Path(__file__).resolve().parent;V10=CB.parent;sys.path.insert(0,str(V10));sys.path.insert(0,str(CB));sys.path.insert(0,str(V10.parent/'expanded_v6_laplace'));os.environ['TIERB_OUT']=str(V10)
import unimol_laplace as M
import tierb_laplace as L
import numpy as np, torch
t,e=L.t,L.e
out=json.loads((CB/'control_no_posterior.json').read_text()) if (CB/'control_no_posterior.json').exists() else {}
for d in sys.argv[1:] or ['freesolv','esol','bace','clintox','bbbp','lipo','qm7','tox21']:
    if d in out:continue
    c=M.cb.UCtx('scaffold',d);dist=L.distances('scaffold',d,['test'])['test'];rows=[]
    for s in M.cb.SINGLE:
        net=M.cb.fit(c,'det',s)[0]
        with np.load(CB/'results'/'scaffold'/d/f'seed-{s}'/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
        mu,v,noise=M.part_outputs(c,net,arr,'test');pm,pv=L.compose(c,mu,torch.zeros_like(v),noise,dist,None,0.);mean,var=L._convert(c,pm,pv)
        nll_nopost=L.ev.macro(c,mean,var)['nll']
        det=[r for r in json.loads((V10/'results'/'Uni-Mol'/'scaffold'/d/'per_seed_metrics.json').read_text()) if r['method']=='Deterministic' and r['seed']==s and r.get('subgroup','all')=='all'][0]['nll']
        lap=[r for r in json.loads((CB/'results'/'scaffold'/d/'per_seed_metrics.json').read_text()) if r['method']=='Laplace' and r['seed']==s and r.get('subgroup','all')=='all'][0]['nll']
        rows.append({'seed':s,'nll_deterministic':det,'nll_no_posterior':nll_nopost,'nll_laplace':lap})
    out[d]=rows;(CB/'control_no_posterior.json').write_text(json.dumps(out,indent=1));print(d,[round(r['nll_no_posterior']-r['nll_deterministic'],4) for r in rows],[round(r['nll_laplace']-r['nll_no_posterior'],4) for r in rows],flush=True)
