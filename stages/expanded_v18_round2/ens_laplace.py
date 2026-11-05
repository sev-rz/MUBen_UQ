"""C4 runner (PROTOCOL.md): Laplace on every ensemble member. E_D = ten Deterministic networks pooled as in the pipeline, E_L = ten Laplace members pooled, E_S = ten recalibrated members pooled;
per-molecule paired NLL with a bootstrap over test molecules (2000 resamples shared by all arms), scaffold split.  python ens_laplace.py BACKBONE [datasets...]"""
import sys,os,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from r2_common import *
bb=sys.argv[1];datasets=sys.argv[2:];S=setup(bb);L=S.L;torch=S.torch;B=2000
MEM=list(range(10));out=D/'results_ens'/bb;out.mkdir(parents=True,exist_ok=True)
def stored_ens(d):
    for p in [V4/'results'/bb/'scaffold'/d/'per_seed_metrics.json',V8/'results'/'GIN'/'scaffold'/d/'per_seed_metrics.json',V10/'results'/'Uni-Mol'/'scaffold'/d/'per_seed_metrics.json',V10/'results'/bb/'scaffold'/d/'per_seed_metrics.json']:
        if p.exists() and (bb!='GIN' or 'GIN' in str(p)) and (bb!='Uni-Mol' or 'Uni-Mol' in str(p)) and (bb not in ('DNN','ChemBERTa') or str(V4) in str(p)):
            r=[x for x in json.loads(p.read_text()) if x['method']=='Ensembles' and x.get('subgroup','all')=='all']
            if r:return r[0]['nll']
    return None
def pool(parts,reg):
    mean=np.mean([a for a,_ in parts],0);return mean,(np.mean([b for _,b in parts],0) if reg else None)
for d in (datasets or DS):
    f=out/f'{d}.json'
    if f.exists():continue
    c=S.mk('scaffold',d);n=int(c.raw_test.shape[0]);reg=c.reg;PD=[];PL=[];PS=[];tick=time.perf_counter()
    for s in MEM:
        assert (S.folder(c,'det',s)/'summary.json').exists(),('network missing',bb,d,s)
        arr,rep=S.fit_group(c,s);net=S.newnet(c,s);mu,v,noise=S.po(c,net,arr,'test')
        pm,pv=L.compose(c,mu,v,noise,torch.zeros(n),None,0.);PL.append(L._convert(c,pm,pv));PD.append(S.det(c,net))
        mu_v,_,noise_v=S.po(c,net,arr,'valid');par=scale_params(S,c,mu_v,noise_v);sm,sv=scaled(S,c,mu,noise,par);PS.append(L._convert(c,sm,sv))
    arms={'E_D':pool(PD,reg),'E_L':pool(PL,reg),'E_S':pool(PS,reg)};met={k:L.ev.macro(c,*v) for k,v in arms.items()}
    y=c.raw_test.astype(float);m=c.mask_test.astype(bool);ok=scored_tasks(y,m,reg);nl={k:per_mol_nll(v[0],v[1],y,m,reg)[0] for k,v in arms.items()}
    rng=np.random.default_rng(int(hashlib.md5(f'{bb}-{d}'.encode()).hexdigest()[:8],16));idx=rng.integers(0,n,(B,n));res={}
    for name,(a,b_) in {'LD':('E_L','E_D'),'LS':('E_L','E_S'),'SD':('E_S','E_D')}.items():
        diff=nl[a]-nl[b_];est=macro_nll(diff,m,ok);boot=macro_nll(diff,m,ok,idx);lo,hi=np.percentile(boot,[2.5,97.5]);se=float(boot.std())
        ident=abs(est)<1e-4 and se<1e-4;z=0.0 if ident else (est/se if se>1e-12 else 0.0)
        res[name]={'est':est,'lo':float(lo),'hi':float(hi),'se':se,'z':float(z),'verdict':'identical' if ident else ('improves' if hi<0 else ('worsens' if lo>0 else 'unclear'))}
    st=stored_ens(d);chk=abs(met['E_D']['nll']-st) if st is not None else None
    f.write_text(json.dumps({'backbone':bb,'dataset':d,'metrics':met,'diffs':res,'stored_ensemble_nll':st,'check_E_D_vs_stored':chk,'macro_from_arrays_E_D':macro_nll(nl['E_D'],m,ok),'seconds':time.perf_counter()-tick}))
    print(json.dumps({'bb':bb,'d':d,'E_L-E_D':round(res['LD']['est'],4),'E_L-E_S':round(res['LS']['est'],4),'E_S-E_D':round(res['SD']['est'],4),'verdict_LD':res['LD']['verdict'],'chk':chk,'s':round(time.perf_counter()-tick,1)}),flush=True)
print('DONE',bb,flush=True)
