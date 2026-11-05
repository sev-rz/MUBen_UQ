"""One seed of one dataset (linear-schedule Deterministic ChemBERTa) plus its test metrics; usage: run_lin_seed.py DATASET SEED. The three seed files are merged into metrics/lin_det_<d>.json by merge_lin.py."""
import os,sys,json
from pathlib import Path
D=Path(__file__).resolve().parent;os.environ['TIERB_OUT']=str(D);sys.path.insert(0,str(D.parent/'expanded_v4'));sys.path.insert(0,str(D))
import tierb_chemberta_linear as cb
import tierb_eval as ev
d,s=sys.argv[1],int(sys.argv[2]);f=D/'metrics_seed'/f'{d}_{s}.json'
if not f.exists():
    c=cb.CCtx('scaffold',d);net,sm=cb.fit(c,'det',s);mu,var=ev.convert(c,cb.predict(net,c.tokens['test']));m=ev.macro(c,mu,var);m.update(dataset=d,seed=s,epochs=sm['epochs'],seconds=sm['seconds'])
    f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(m));print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items() if k in ('dataset','seed','nll','epochs','seconds')}),flush=True)
