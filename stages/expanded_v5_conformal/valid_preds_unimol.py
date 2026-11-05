"""Validation-split predictions of the Uni-Mol reproduction for conformal calibration (no retraining; test split never touched). Mirrors valid_preds_gin.py.
  python valid_preds_unimol.py [--datasets d ...]    (authors' environment python, GPU)"""
import sys,os,json,argparse
from pathlib import Path
W=Path(__file__).resolve().parent.parent;V10=W/'expanded_v10_unimol';sys.path.insert(0,str(V10));os.environ['TIERB_OUT']=str(V10)
import tierb_unimol as u
import tierb_unimol_ext as x
t,e=u.t,u.e
import numpy as np
OUT=Path(__file__).resolve().parent/'preds'/'Uni-Mol'
def to_valid(c):
    tt=t.table(c.d);v=t.split_index(c.d,c.split)['valid'];c._cache['test']=c.eval_batches('valid')
    c.raw_test=tt['y'][v].astype(float);c.mask_test=tt['m'][v];c.ids=dict(c.ids);c.ids['test']=c.ids['valid'];assert len(c.raw_test)==len(v)
def run(datasets):
    for d in datasets:
        o=OUT/'scaffold'/d
        if (o/'DONE').exists():continue
        if not ((V10/'frozen'/f'Uni-Mol_scaffold_{d}.json').exists() and (V10/'frozen_ext'/f'Uni-Mol_scaffold_{d}.json').exists()):print('not frozen (base or ext), skip',d,flush=True);continue
        c=u.UCtx('scaffold',d);to_valid(c);o.mkdir(parents=True,exist_ok=True);n=0
        for (m,s),(mu,var) in x.predictions_all(c).items():
            a=dict(mean=mu,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
            if var is not None:a['variance']=var
            np.savez_compressed(o/f'{m.replace(" ","_")}-{s}.npz',**a);n+=1
        (o/'DONE').write_text(json.dumps({'files':n}));print('VALID Uni-Mol',d,n,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args();run(a.datasets)
