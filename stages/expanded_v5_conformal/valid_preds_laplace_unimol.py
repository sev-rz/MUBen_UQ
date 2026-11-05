"""Validation predictions (from the saved posteriors, test never touched) and staged TEST copies for the Uni-Mol Laplace family, for conformal calibration. Mirrors valid_preds_laplace.py (GIN branch).
Outputs: preds/Uni-Mol-Laplace/scaffold/<dataset>/<Method>-<seed>.npz (validation, plus DONE) and preds_test/Uni-Mol-Laplace/scaffold/... (copies of the test predictions of the Laplace stage).
  python valid_preds_laplace_unimol.py"""
import sys,os,json,shutil
from pathlib import Path
W=Path(__file__).resolve().parent.parent;C=Path(__file__).resolve().parent;V10=W/'expanded_v10_unimol';V6=W/'expanded_v6_laplace'
os.environ['TIERB_OUT']=str(V10);sys.path.insert(0,str(V6));sys.path.insert(0,str(V10));sys.path.insert(0,str(V10/'laplace'))
import unimol_laplace as M
import tierb_laplace as L
import numpy as np, torch
t,e=L.t,L.e
res=V10/'laplace'/'results';fro=V10/'laplace'/'frozen';OUT=C/'preds'/'Uni-Mol-Laplace';OUTT=C/'preds_test'/'Uni-Mol-Laplace';split='scaffold'
for d in t.DATASETS:
    o=OUT/split/d
    if (o/'DONE').exists() or not ((fro/f'Uni-Mol_{split}_{d}.json').exists() and (fro/f'NOV_Uni-Mol_{split}_{d}.json').exists()):continue
    c=M.cb.UCtx(split,d);tt=t.table(d);v=t.split_index(d,split)['valid'];y=tt['y'][v].astype(float);m=tt['m'][v];ids=c.ids['valid'];dist=L.distances(split,d,['valid'])['valid'];o.mkdir(parents=True,exist_ok=True);n=0
    for s in M.cb.SINGLE:
        net=M.cb.fit(c,'det',s)[0]
        with np.load(res/split/d/f'seed-{s}'/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
        novj=json.loads((res/split/d/f'seed-{s}'/'novelty.json').read_text())['variants'];mu,vv,noise=M.part_outputs(c,net,arr,'valid')
        preds={'Laplace':L.compose(c,mu,vv,noise,dist,None,0.)}
        for var in L.VARIANTS:preds[var]=L.compose(c,mu,vv,noise,dist,var,torch.tensor([r['lambda'] for r in novj[var]],dtype=torch.float64))
        for name,(pm,pv) in preds.items():
            mean,variance=L._convert(c,pm,pv);a=dict(mean=mean,labels=y,masks=m,ids=ids)
            if variance is not None:a['variance']=variance
            np.savez_compressed(o/f'{name.replace(" ","_")}-{s}.npz',**a);n+=1
    (o/'DONE').write_text(json.dumps({'files':n}));tdir=OUTT/split/d;tdir.mkdir(parents=True,exist_ok=True)
    for f in (res/split/d).glob('*.npz'):shutil.copy2(f,tdir/f.name)
    print('VALID Uni-Mol Laplace',split,d,n,flush=True)
