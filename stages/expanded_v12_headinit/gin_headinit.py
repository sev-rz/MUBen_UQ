"""Head-initialisation test (n0111 follow-up): our GIN pipeline with the output head at PyTorch default init (the authors' OutputLayer) instead of Xavier + bias 0.01. Deterministic, scaffold, seeds 0-9, test metrics with the usual tierb_eval macro."""
import os,sys,json
os.environ['OMP_NUM_THREADS']=os.environ.get('OMP_NUM_THREADS','4')
from pathlib import Path
D=Path(__file__).resolve().parent;os.environ['TIERB_OUT']=str(D);sys.path.insert(0,str(D))
import tierb_gin_headinit as g
t,e=g.t,g.e
import tierb_eval as ev
import numpy as np
def run(d,seeds):
    c=g.GCtx('scaffold',d);out=[]
    for s in seeds:
        net,_=g.fit(c,'det',s);mu,var=ev.convert(c,g.predict(net,c.graphs['test']));m=ev.macro(c,mu,var);m.update(dataset=d,seed=s,method='Deterministic');out.append(m);print(json.dumps(m),flush=True)
        (D/'metrics').mkdir(exist_ok=True);(D/'metrics'/f'GIN_{d}.json').write_text(json.dumps(out))
if __name__=='__main__':
    d=sys.argv[1];run(d,[int(x) for x in sys.argv[2:]] or list(range(10)))
