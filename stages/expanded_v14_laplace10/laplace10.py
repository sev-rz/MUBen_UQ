"""Item B runner (PROTOCOL.md): per network last-layer Laplace (unchanged stage code) and Deterministic test metrics, scaffold split.
  python laplace10.py BACKBONE [datasets...]     BACKBONE in DNN | ChemBERTa | GIN | Uni-Mol"""
import os,sys,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;bb=sys.argv[1];datasets=sys.argv[2:]
V4=W/'expanded_v4';V6=W/'expanded_v6_laplace';V8=W/'expanded_v8_gin';V10=W/'expanded_v10_unimol'
sys.path.insert(0,str(V4));sys.path.insert(0,str(V6))
if bb=='GIN':os.environ['TIERB_OUT']=str(V8);sys.path.insert(0,str(V8));sys.path.insert(0,str(V8/'laplace'))
if bb=='Uni-Mol':os.environ['TIERB_OUT']=str(V10);sys.path.insert(0,str(V10));sys.path.insert(0,str(V10/'laplace'))
if bb=='ChemBERTa':sys.path.insert(0,str(V6/'chemberta'))
import numpy as np,torch
if bb=='DNN':
    import tierb_laplace as L;t=L.t;mk=t.Ctx;fit_group=L.fit_group;po=L.part_outputs;folder=t.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,t.e.predict(net,c.test[0]))
elif bb=='ChemBERTa':
    import chemberta_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.CCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.tokens['test']))
elif bb=='GIN':
    import gin_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.GCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.graphs['test']))
else:
    import unimol_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.UCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.agg(cb.predict_raw(net,c,'test'),c.reg))
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7']
SEEDS=list(range(10))+([] if bb=='Uni-Mol' else [100,200])
out=D/'results'/bb;out.mkdir(parents=True,exist_ok=True)
for d in (datasets or DS):
    f=out/f'{d}.json'
    if f.exists():continue
    c=mk('scaffold',d);rows=[];n=int(c.raw_test.shape[0])
    for s in SEEDS:
        assert (folder(c,'det',s)/'summary.json').exists(),('network missing (would be retrained)',bb,d,s)
        tick=time.perf_counter();arr,rep=fit_group(c,s);net=folder and None
        net=(t.fit(c,'det',s)[0] if bb=='DNN' else cb.fit(c,'det',s)[0])
        mu,v,noise=po(c,net,arr,'test');pm,pv=L.compose(c,mu,v,noise,torch.zeros(n),None,0.);mean,var=L._convert(c,pm,pv);mL=L.ev.macro(c,mean,var)
        md_,vd_=det(c,net);mD=L.ev.macro(c,md_,vd_)
        edge=sum(r['alpha'] in (L.ALPHAS[0],L.ALPHAS[-1]) for r in rep['per_task'])
        rows.append({'backbone':bb,'dataset':d,'seed':s,'laplace':{k:v_ for k,v_ in mL.items()},'deterministic':{k:v_ for k,v_ in mD.items()},'alpha_edge_tasks':edge,'tasks':len(rep['per_task']),'seconds':time.perf_counter()-tick})
        print(json.dumps({'bb':bb,'d':d,'seed':s,'dNLL':round(mL['nll']-mD['nll'],4),'s':round(time.perf_counter()-tick,1)}),flush=True)
    f.write_text(json.dumps(rows))
print('DONE',bb,flush=True)
