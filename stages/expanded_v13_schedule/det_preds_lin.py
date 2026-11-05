"""Save per-molecule test predictions of the linear-schedule Deterministic nets (for the check-E statistics): laplace/results/scaffold/<d>/Deterministic-<seed>.npz (mean, variance, labels, masks, ids)."""
import os,sys,json
from pathlib import Path
D=Path(__file__).resolve().parent;os.environ['TIERB_OUT']=str(D);sys.path.insert(0,str(D.parent/'expanded_v4'));sys.path.insert(0,str(D))
import tierb_chemberta_linear as cb
import tierb_eval as ev
import numpy as np
for d in cb.t.DATASETS:
    c=cb.CCtx('scaffold',d);o=D/'laplace'/'results'/'scaffold'/d;o.mkdir(parents=True,exist_ok=True)
    for s in [0,100,200]:
        f=o/f'Deterministic-{s}.npz'
        if f.exists():continue
        net,_=cb.fit(c,'det',s);mu,var=ev.convert(c,cb.predict(net,c.tokens['test']));a=dict(mean=mu,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
        if var is not None:a['variance']=var
        np.savez_compressed(f,**a)
    print('saved',d,flush=True)
