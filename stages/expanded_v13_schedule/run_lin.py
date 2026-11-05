"""Deterministic linear-schedule ChemBERTa fits (seeds 0/100/200) and their test metrics; usage: run_lin.py DATASET...  (processes may run in parallel on different datasets)."""
import os,sys,json
from pathlib import Path
D=Path(__file__).resolve().parent;os.environ['TIERB_OUT']=str(D);sys.path.insert(0,str(D.parent/'expanded_v4'));sys.path.insert(0,str(D))
import tierb_chemberta_linear as cb
import tierb_eval as ev
for d in sys.argv[1:]:
    mf=D/'metrics'/f'lin_det_{d}.json'
    if mf.exists():continue
    c=cb.CCtx('scaffold',d);rows=[]
    for s in [0,100,200]:
        net,sm=cb.fit(c,'det',s);mu,var=ev.convert(c,cb.predict(net,c.tokens['test']));m=ev.macro(c,mu,var);m.update(dataset=d,seed=s,epochs=sm['epochs'],seconds=sm['seconds']);rows.append(m);print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items() if k in ('dataset','seed','nll','epochs','seconds','rmse','roc-auc')}),flush=True)
    mf.parent.mkdir(exist_ok=True);mf.write_text(json.dumps(rows))
