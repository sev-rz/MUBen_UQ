"""Validation-split predictions of the GIN reproduction for conformal calibration (no retraining; test split never touched). Mirrors valid_preds.py.
  python valid_preds_gin.py [splits...] [--datasets d ...]    (run with the authors' environment python, GPU)"""
import sys,os,json,argparse
from pathlib import Path
W=Path(__file__).resolve().parent.parent;V8=W/'expanded_v8_gin';sys.path.insert(0,str(V8));os.environ['TIERB_OUT']=str(V8)
import tierb_gin as g
t,e=g.t,g.e
import tierb_eval as ev
import numpy as np
OUT=Path(__file__).resolve().parent/'preds'/'GIN'
def to_valid(c):
    tt=t.table(c.d);v=t.split_index(c.d,c.split)['valid'];c.graphs=dict(c.graphs);c.graphs['test']=c.graphs['valid']
    c.raw_test=tt['y'][v].astype(float);c.mask_test=tt['m'][v];c.ids=dict(c.ids);c.ids['test']=c.ids['valid'];assert len(c.raw_test)==len(v)
def run(splits,datasets):
    for split in splits:
        for d in datasets:
            o=OUT/split/d
            if (o/'DONE').exists():continue
            if not (V8/'frozen'/f'GIN_{split}_{d}.json').exists():print('not frozen, skip',split,d,flush=True);continue
            c=g.GCtx(split,d);to_valid(c);o.mkdir(parents=True,exist_ok=True);n=0
            for (m,s),(mu,var) in g.predictions_gin(c).items():
                a=dict(mean=mu,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
                if var is not None:a['variance']=var
                np.savez_compressed(o/f'{m.replace(" ","_")}-{s}.npz',**a);n+=1
            (o/'DONE').write_text(json.dumps({'files':n}));print('VALID GIN',split,d,n,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('splits',nargs='*',default=t.SPLITS);p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args();run(a.splits,a.datasets)
