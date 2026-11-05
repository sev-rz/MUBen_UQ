"""Validation (and test, for format) predictions of the phase-2 Laplace family with the unchanged phase-2 code and saved
posteriors/corrections (no refitting). Test predictions are asserted identical to expanded_v2/results/*.npz."""
import sys,os,json
from pathlib import Path
import numpy as np
C=Path(__file__).resolve().parent;V2=C.parent/'expanded_v2';sys.path.insert(0,str(V2));os.chdir(V2)
import run_expansion as rx
from scipy.special import expit
METHODS=['Deterministic','Laplace','Laplace constant','Novelty only','Novelty Laplace']
for d in ['bace','bbbp','esol']:
    _,_,ts,_=rx.data(d,'train');reg=d=='esol'
    for part,root in [('valid',C/'preds'),('test',C/'preds_test')]:
        px,py,ps,ids=rx.data(d,part);dist=rx.novelty(ts,ps);out=root/'DNN-v2'/'scaffold'/d;out.mkdir(parents=True,exist_ok=True)
        for group in rx.GROUPS:
            s=group[0];m,meta=rx.train(d,s);folder=rx.ROOT/'fits'/d/f'seed-{s}';fit=json.loads((folder/'settings.json').read_text())
            o=rx.predict(m,px);raw=o[:,0].numpy().astype(float) if reg else o.numpy().astype(float).ravel()
            base=rx.noise(o).numpy().astype(float) if reg else np.zeros(len(raw))
            pred={'Deterministic':(raw,base) if reg else (expit(raw),None)}
            with np.load(folder/'laplace_posterior.npz') as z:cov=z['covariance']
            th=rx.phi(rx.features(m,px)).numpy();v=np.maximum(np.einsum('ij,jk,ik->i',th,cov,th),0)
            pred['Laplace']=rx.integrate(raw,base,v,reg)
            for method,setting in fit['corrections'].items():
                initial=np.zeros_like(v) if method=='Novelty only' else v;factor=np.ones_like(v) if method=='Laplace constant' else dist**2
                pred[method]=rx.integrate(raw,base,initial+setting['selected_lambda']*factor,reg)
            y=py.numpy().astype(float).reshape(-1,1)
            for name in METHODS:
                mu,var=pred[name];mu=np.asarray(mu,float).reshape(-1)
                if reg:mu=mu*meta['target_std']+meta['target_mean'];var=np.asarray(var,float).reshape(-1)*meta['target_std']**2
                if part=='test':
                    ref=np.load(V2/'results'/f'{d}-{s}-{name.replace(" ","_")}.npz')
                    np.testing.assert_allclose(mu,ref['mean'].reshape(-1),atol=1e-9,rtol=0)
                a=dict(mean=mu.reshape(-1,1),labels=y,masks=np.ones_like(y,bool),ids=np.asarray(ids))
                if reg:a['variance']=var.reshape(-1,1)
                np.savez_compressed(out/f'{name.replace(" ","_")}-{s}.npz',**a)
        (out/'DONE').write_text('ok');print('DONE',part,d,flush=True)
