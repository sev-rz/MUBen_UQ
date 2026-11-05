"""C3 runner (PROTOCOL.md): evidence-based prior strength versus validation-tuned alpha, Deterministic and recalibration, per saved network, scaffold split.
  python evidence.py BACKBONE [datasets...]    env NSEEDS=n limits seeds (smoke test only; results go to results_smoke)"""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from r2_common import *
bb=sys.argv[1];datasets=sys.argv[2:];S=setup(bb);L=S.L;torch=S.torch
NS=int(os.environ.get('NSEEDS','0'));SEEDS=S.seeds[:NS] if NS else S.seeds
out=D/('results_smoke' if NS else 'results_evidence')/bb;out.mkdir(parents=True,exist_ok=True)
for d in (datasets or DS):
    f=out/f'{d}.json'
    if f.exists():continue
    c=S.mk('scaffold',d);rows=[];n=int(c.raw_test.shape[0]);prev={r['seed']:r for r in json.loads((V14/'results'/bb/f'{d}.json').read_text())}
    for s in SEEDS:
        assert (S.folder(c,'det',s)/'summary.json').exists(),('network missing',bb,d,s)
        tick=time.perf_counter();CAP,orig=capture(L)
        try:arrN,repN=S.fit_group(c,s)
        finally:L.cov_from=orig
        assert len(CAP)==c.T,(len(CAP),c.T)
        th=torch.from_numpy(arrN['theta']);aev=[];covs=[]
        for j in range(c.T):
            a=evidence_alpha(torch,CAP[j],th[j]);aev.append(a);covs.append(orig(CAP[j],a))
        arrE=dict(arrN);arrE['covariance']=torch.stack(covs).numpy()
        net=S.newnet(c,s);mu,v,noise=S.po(c,net,arrN,'test')
        pm,pv=L.compose(c,mu,v,noise,torch.zeros(n),None,0.);mean,var=L._convert(c,pm,pv);mL=L.ev.macro(c,mean,var)
        mu2,v2,noise2=S.po(c,net,arrE,'test');pm,pv=L.compose(c,mu2,v2,noise2,torch.zeros(n),None,0.);mean,var=L._convert(c,pm,pv);mE=L.ev.macro(c,mean,var)
        md,vd=S.det(c,net);mD=L.ev.macro(c,md,vd)
        mu_v,_,noise_v=S.po(c,net,arrN,'valid');par=scale_params(S,c,mu_v,noise_v);sm,sv=scaled(S,c,mu,noise,par);mean_s,var_s=L._convert(c,sm,sv);mS=L.ev.macro(c,mean_s,var_s)
        ap=[r['alpha'] for r in repN['per_task']];edge_p=sum(a in (L.ALPHAS[0],L.ALPHAS[-1]) for a in ap);edge_e=sum(a in (EGRID[0],EGRID[-1]) for a in aev)
        p=prev.get(s);diff=abs(mL['nll']-p['laplace']['nll']) if p else None
        rows.append({'backbone':bb,'dataset':d,'seed':s,'laplace':mL,'evidence':mE,'scaling':mS,'deterministic':mD,'alpha_primary':ap,'alpha_evidence':aev,'edge_primary':edge_p,'edge_evidence':edge_e,'tasks':c.T,
                     'repro_diff_vs_itemB':diff,'seconds':time.perf_counter()-tick})
        print(json.dumps({'bb':bb,'d':d,'seed':s,'L-D':round(mL['nll']-mD['nll'],4),'E-D':round(mE['nll']-mD['nll'],4),'S-D':round(mS['nll']-mD['nll'],4),'dlog10a':round(float(np.median(np.log10(aev)-np.log10(ap))),2),'repro':diff,'s':round(time.perf_counter()-tick,1)}),flush=True)
    f.write_text(json.dumps(rows))
print('DONE',bb,flush=True)
