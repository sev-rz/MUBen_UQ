"""Item B1/B2 runner (PROTOCOL.md): per network Laplace primary grid, Laplace wide grid, one-parameter recalibration baseline, Deterministic. Scaffold split.
  python wide_scale.py BACKBONE [datasets...]     env NSEEDS=n limits the number of seeds (smoke test only; results then go to results_smoke)"""
import os,sys,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;bb=sys.argv[1];datasets=sys.argv[2:]
V4=W/'expanded_v4';V6=W/'expanded_v6_laplace';V8=W/'expanded_v8_gin';V10=W/'expanded_v10_unimol';V14=W/'expanded_v14_laplace10'
sys.path.insert(0,str(V4));sys.path.insert(0,str(V6))
if bb=='GIN':os.environ['TIERB_OUT']=str(V8);sys.path.insert(0,str(V8));sys.path.insert(0,str(V8/'laplace'))
if bb=='Uni-Mol':os.environ['TIERB_OUT']=str(V10);sys.path.insert(0,str(V10));sys.path.insert(0,str(V10/'laplace'))
if bb=='ChemBERTa':sys.path.insert(0,str(V6/'chemberta'))
import numpy as np,torch
M=None
if bb=='DNN':
    import tierb_laplace as L;t=L.t;mk=t.Ctx;fit_group=L.fit_group;po=L.part_outputs;folder=t.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,t.e.predict(net,c.test[0]))
    newnet=lambda c,s:t.fit(c,'det',s)[0]
elif bb=='ChemBERTa':
    import chemberta_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.CCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.tokens['test']))
    newnet=lambda c,s:cb.fit(c,'det',s)[0]
elif bb=='GIN':
    import gin_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.GCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.graphs['test']))
    newnet=lambda c,s:cb.fit(c,'det',s)[0]
else:
    import unimol_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.UCtx;fit_group=M.fit_group;po=M.part_outputs;folder=cb.folder
    import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.agg(cb.predict_raw(net,c,'test'),c.reg))
    newnet=lambda c,s:cb.fit(c,'det',s)[0]
NARROW=list(L.ALPHAS);WIDE=[10.**a for a in range(-8,5)];_orig=L.cov_from;FALL=[0]
def safe(H,alpha):
    A=H+alpha*torch.eye(len(H),dtype=H.dtype)
    try:return torch.cholesky_inverse(torch.linalg.cholesky(A))
    except Exception:
        FALL[0]+=1;w,V=torch.linalg.eigh(A);return (V/w.clamp_min(alpha))@V.T
def use(grid,cov):
    L.ALPHAS=grid;L.cov_from=cov
    if M is not None:M.ALPHAS=grid
TGRID=torch.logspace(np.log10(0.05),np.log10(20.),81,dtype=torch.float64)
def scale_params(c,mu,noise):
    """Per task, VALIDATION data only: regression variance multiplier (closed-form ML), classification temperature (grid)."""
    y=c.valid[1].double();m=c.valid[2].double();out=[]
    for j in range(c.T):
        k=m[:,j].bool()
        if c.reg:out.append(float(((y[k,j]-mu[k,j])**2/noise[k,j].clamp_min(1e-6)).mean()))
        else:
            z=mu[k,j];yy=y[k,j];sc=[L.nll_cls(yy,torch.sigmoid(z/T)) for T in TGRID];out.append(float(TGRID[int(np.argmin(sc))]))
    return torch.tensor(out,dtype=torch.float64)
def scaled(c,mu,noise,par):
    return (mu,noise*par[None,:]) if c.reg else (torch.sigmoid(mu/par[None,:]),None)
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7']
NS=int(os.environ.get('NSEEDS','0'));SEEDS=list(range(10))+([] if bb=='Uni-Mol' else [100,200])
if NS:SEEDS=SEEDS[:NS]
out=D/('results_smoke' if NS else 'results')/bb;out.mkdir(parents=True,exist_ok=True)
for d in (datasets or DS):
    f=out/f'{d}.json'
    if f.exists():continue
    c=mk('scaffold',d);rows=[];n=int(c.raw_test.shape[0]);prev={r['seed']:r for r in json.loads((V14/'results'/bb/f'{d}.json').read_text())}
    for s in SEEDS:
        assert (folder(c,'det',s)/'summary.json').exists(),('network missing (would be retrained)',bb,d,s)
        tick=time.perf_counter()
        use(NARROW,_orig);arrN,repN=fit_group(c,s);net=newnet(c,s)
        mu,v,noise=po(c,net,arrN,'test');pm,pv=L.compose(c,mu,v,noise,torch.zeros(n),None,0.);mean,var=L._convert(c,pm,pv);mL=L.ev.macro(c,mean,var)
        md_,vd_=det(c,net);mD=L.ev.macro(c,md_,vd_)
        mu_v,_,noise_v=po(c,net,arrN,'valid');par=scale_params(c,mu_v,noise_v)
        sm,sv=scaled(c,mu,noise,par);mean_s,var_s=L._convert(c,sm,sv);mS=L.ev.macro(c,mean_s,var_s)
        use(WIDE,safe);arrW,repW=fit_group(c,s)
        mu,v,noise=po(c,net,arrW,'test');pm,pv=L.compose(c,mu,v,noise,torch.zeros(n),None,0.);mean,var=L._convert(c,pm,pv);mW=L.ev.macro(c,mean,var)
        use(NARROW,_orig)
        edgeN=sum(r['alpha'] in (NARROW[0],NARROW[-1]) for r in repN['per_task']);edgeW=sum(r['alpha'] in (WIDE[0],WIDE[-1]) for r in repW['per_task'])
        p=prev.get(s);diff=abs(mL['nll']-p['laplace']['nll']) if p else None
        rows.append({'backbone':bb,'dataset':d,'seed':s,'laplace':mL,'laplace_wide':mW,'scaling':mS,'deterministic':mD,'scale_params':[float(x) for x in par],
                     'alpha_primary':[r['alpha'] for r in repN['per_task']],'alpha_wide':[r['alpha'] for r in repW['per_task']],'edge_primary':edgeN,'edge_wide':edgeW,'tasks':len(repN['per_task']),
                     'repro_diff_vs_itemB':diff,'chol_fallbacks_total':FALL[0],'seconds':time.perf_counter()-tick})
        print(json.dumps({'bb':bb,'d':d,'seed':s,'L-D':round(mL['nll']-mD['nll'],4),'S-D':round(mS['nll']-mD['nll'],4),'Lw-L':round(mW['nll']-mL['nll'],4),'repro':diff,'edge':[edgeN,edgeW],'s':round(time.perf_counter()-tick,1)}),flush=True)
    f.write_text(json.dumps(rows))
print('DONE',bb,flush=True)
