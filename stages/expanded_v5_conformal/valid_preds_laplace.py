"""Validation predictions (from the saved posteriors, test never touched) and staged TEST copies for the Laplace family of one backbone, for conformal calibration.
Outputs: preds/<BB>-Laplace/<split>/<dataset>/<Method>-<seed>.npz (validation, plus DONE) and preds_test/<BB>-Laplace/... (copies of the test predictions made by the Laplace stages).
  python valid_preds_laplace.py DNN|ChemBERTa|GIN"""
import sys,os,json,shutil
from pathlib import Path
bb=sys.argv[1];W=Path(__file__).resolve().parent.parent;C=Path(__file__).resolve().parent;V4=W/'expanded_v4';V6=W/'expanded_v6_laplace';V8=W/'expanded_v8_gin'
sys.path.insert(0,str(V6))
if bb=='GIN':os.environ['TIERB_OUT']=str(V8);sys.path.insert(0,str(V8/'laplace'));import gin_laplace as M
elif bb=='ChemBERTa':sys.path.insert(0,str(V6/'chemberta'));import chemberta_laplace as M
else:import tierb_laplace as M
import tierb_laplace as L
import numpy as np, torch
t,e=L.t,L.e
if bb=='DNN':mk=t.Ctx;res=V6/'results';fro=V6/'frozen';getnet=lambda c,s:t.fit(c,'det',s)[0];po=L.part_outputs;prim='DNN_{}_{}.json';nov='NOV_{}_{}.json'
elif bb=='ChemBERTa':mk=M.cb.CCtx;res=V6/'chemberta'/'results';fro=V6/'chemberta'/'frozen';getnet=lambda c,s:M.cb.fit(c,'det',s)[0];po=M.part_outputs;prim='ChemBERTa_{}_{}.json';nov='NOV_ChemBERTa_{}_{}.json'
else:mk=M.cb.GCtx;res=V8/'laplace'/'results';fro=V8/'laplace'/'frozen';getnet=lambda c,s:M.cb.fit(c,'det',s)[0];po=M.part_outputs;prim='GIN_{}_{}.json';nov='NOV_GIN_{}_{}.json'
OUT=C/'preds'/f'{bb}-Laplace';OUTT=C/'preds_test'/f'{bb}-Laplace'
for split in t.SPLITS:
    for d in t.DATASETS:
        o=OUT/split/d
        if (o/'DONE').exists() or not ((fro/prim.format(split,d)).exists() and (fro/nov.format(split,d)).exists()):continue
        c=mk(split,d);tt=t.table(d);v=t.split_index(d,split)['valid'];y=tt['y'][v].astype(float);m=tt['m'][v];ids=c.ids['valid'];dist=L.distances(split,d,['valid'])['valid'];o.mkdir(parents=True,exist_ok=True);n=0
        for s in t.seeds(split)[0]:
            net=getnet(c,s)
            with np.load(res/split/d/f'seed-{s}'/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
            novj=json.loads((res/split/d/f'seed-{s}'/'novelty.json').read_text())['variants'];mu,vv,noise=po(c,net,arr,'valid')
            preds={'Laplace':L.compose(c,mu,vv,noise,dist,None,0.)}
            for var in L.VARIANTS:preds[var]=L.compose(c,mu,vv,noise,dist,var,torch.tensor([r['lambda'] for r in novj[var]],dtype=torch.float64))
            for name,(pm,pv) in preds.items():
                mean,variance=L._convert(c,pm,pv);a=dict(mean=mean,labels=y,masks=m,ids=ids)
                if variance is not None:a['variance']=variance
                np.savez_compressed(o/f'{name.replace(" ","_")}-{s}.npz',**a);n+=1
        (o/'DONE').write_text(json.dumps({'files':n}));tdir=OUTT/split/d;tdir.mkdir(parents=True,exist_ok=True)
        for f in (res/split/d).glob('*.npz'):shutil.copy2(f,tdir/f.name)
        print('VALID',bb,'Laplace',split,d,n,flush=True)
