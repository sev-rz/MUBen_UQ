"""Validation-split predictions for conformal calibration (no retraining, test split never touched).
Reuses expanded_v4/tierb_eval.py prediction functions by pointing the context's 'test' slots at the validation split.
  python valid_preds.py DNN|ChemBERTa [splits...] [--datasets d ...]"""
import sys,os,json,argparse
from pathlib import Path
import numpy as np
V=Path(__file__).resolve().parent.parent/'expanded_v4'; sys.path.insert(0,str(V)); os.chdir(V)
import tierb_dnn as t, tierb_eval as ev
OUT=Path(__file__).resolve().parent/'preds'
def to_valid(c,backbone):
    tt=t.table(c.d);ii=t.split_index(c.d,c.split);v=ii['valid']
    if backbone=='DNN':c.test=c.valid
    else:c.tokens=dict(c.tokens);c.tokens['test']=c.tokens['valid']
    c.raw_test=tt['y'][v].astype(float);c.mask_test=tt['m'][v];c.ids=dict(c.ids);c.ids['test']=c.ids['valid']
    assert len(c.raw_test)==len(v)
def run(backbone,splits,datasets):
    if backbone=='DNN':mk,pred=t.Ctx,ev.predictions_dnn
    else:
        import tierb_chemberta as cb;mk,pred=cb.CCtx,ev.predictions_chemberta
    for split in splits:
        for d in datasets:
            o=OUT/backbone/split/d
            if (o/'DONE').exists():continue
            if not (t.OUT/'frozen'/f'{backbone}_{split}_{d}.json').exists():print('not frozen, skip',backbone,split,d,flush=True);continue
            c=mk(split,d);to_valid(c,backbone);o.mkdir(parents=True,exist_ok=True);n=0
            for (m,s),(mu,var) in pred(c).items():
                a=dict(mean=mu,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
                if var is not None:a['variance']=var
                np.savez_compressed(o/f'{m.replace(" ","_")}-{s}.npz',**a);n+=1
            (o/'DONE').write_text(json.dumps({'files':n}));print('VALID',backbone,split,d,n,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backbone');p.add_argument('splits',nargs='*',default=t.SPLITS);p.add_argument('--datasets',nargs='*',default=t.DATASETS)
    a=p.parse_args();run(a.backbone,a.splits,a.datasets)
