"""ChemBERTa test arms (n0111 follow-up). arm = linear (our pipeline + linear schedule, 10 % warm-up) | default (our pipeline + PyTorch-default head init). Deterministic, scaffold; usage: chemberta_arm.py ARM DATASET SEEDS..."""
import os,sys,json
from pathlib import Path
D=Path(__file__).resolve().parent;arm=sys.argv[1];d=sys.argv[2];seeds=[int(x) for x in sys.argv[3:]]
os.environ['TIERB_OUT']=str(D/f'out_chemberta_{arm}');sys.path.insert(0,str(D.parent/'expanded_v4'));sys.path.insert(0,str(D))
import importlib;cb=importlib.import_module(f'tierb_chemberta_{arm}')
import tierb_eval as ev
out=[];c=cb.CCtx('scaffold',d)
for s in seeds:
    net,_=cb.fit(c,'det',s);mu,var=ev.convert(c,cb.predict(net,c.tokens['test']));m=ev.macro(c,mu,var);m.update(dataset=d,seed=s,arm=arm);out.append(m);print(json.dumps(m),flush=True)
    (D/'metrics').mkdir(exist_ok=True);(D/'metrics'/f'ChemBERTa_{arm}_{d}.json').write_text(json.dumps(out))
