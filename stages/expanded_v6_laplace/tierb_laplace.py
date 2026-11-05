"""Laplace on the Tier-B DNN (protocol: PROTOCOL.md, sha256 in PROTOCOL.sha256). Reuses the SAVED Tier-B Deterministic nets (tierb_dnn.fit reloads them);
fitting and alpha selection use TRAIN and VALID data only. Test data are not referenced in this file's fit stage.
Classification: independent affine last-layer Bernoulli posterior per task (129 parameters), probit predictive.
Regression: conditional Laplace over the mean head per task, frozen representation and frozen predicted noise, H = Phi^T diag(m/noise) Phi."""
import os, sys, json, time, math, argparse
from pathlib import Path
V6=Path(__file__).resolve().parent; WORK=V6.parent; V4=WORK/'expanded_v4'
sys.path.insert(0,str(V4))
import numpy as np, torch
from torch.nn import functional as F
import tierb_dnn as t
e=t.e
ALPHAS=[10.**a for a in range(-4,5)]
torch.set_num_threads(1)

def feats(net,x):
    net.eval()
    with torch.no_grad():return torch.cat([net.features(b) for b in x.split(256)]).double()
def phi(h):return torch.cat([h,torch.ones((len(h),1),dtype=h.dtype)],1)
def hessian_cls(P,theta,m):
    p=torch.sigmoid(P@theta);return P.T@((m*p*(1-p))[:,None]*P)
def hessian_reg(P,noise,m):return P.T@((m/noise)[:,None]*P)
def cov_from(H,alpha):
    return torch.cholesky_inverse(torch.linalg.cholesky(H+alpha*torch.eye(len(H),dtype=H.dtype)))
def var_of(P,cov):return torch.einsum('ij,jk,ik->i',P,cov,P).clamp_min(0)
def probit(z,v):return torch.sigmoid(z/torch.sqrt(1+math.pi*v/8))
def nll_cls(y,p):
    p=p.clamp(1e-7,1-1e-7);return float(-(y*p.log()+(1-y)*(1-p).log()).mean())
def nll_reg(y,mu,var):return float((0.5*(var.log()+(y-mu)**2/var)).mean())

def checks():
    """Analytic Hessians against autograd, and zero covariance recovers the plain model."""
    g=torch.Generator().manual_seed(7);P=phi(torch.randn(17,4,generator=g,dtype=torch.float64));theta=torch.randn(5,generator=g,dtype=torch.float64)
    y=torch.randint(0,2,(17,),generator=g).double();ones=torch.ones(17,dtype=torch.float64)
    A=torch.autograd.functional.hessian(lambda w:F.binary_cross_entropy_with_logits(P@w,y,reduction='sum'),theta)
    torch.testing.assert_close(A,hessian_cls(P,theta,ones),atol=1e-10,rtol=1e-10)
    noise=torch.rand(17,generator=g,dtype=torch.float64)+.1;yr=torch.randn(17,generator=g,dtype=torch.float64)
    B=torch.autograd.functional.hessian(lambda w:(.5*(yr-P@w)**2/noise).sum(),theta)
    torch.testing.assert_close(B,hessian_reg(P,noise,ones),atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(probit(P@theta,var_of(P,torch.zeros(5,5,dtype=torch.float64))),torch.sigmoid(P@theta))
    mask=ones.clone();mask[:5]=0   # masked rows must not contribute
    torch.testing.assert_close(hessian_cls(P,theta,mask),hessian_cls(P[5:],theta,ones[5:]))
    return {'analytic_hessian_cls_matches_autograd':True,'analytic_hessian_reg_matches_autograd':True,'zero_covariance_recovers_plain':True,'masked_rows_excluded':True}

def fit_group(c,seed):
    """Alpha per task on validation NLL; returns arrays and a per-task report. Uses c.train and c.valid only."""
    tick=time.perf_counter();net,_=t.fit(c,'det',seed);net.eval()
    Ptr=phi(feats(net,c.train[0]));Pva=phi(feats(net,c.valid[0]))
    ytr,mtr=c.train[1].double(),c.train[2].double();yva,mva=c.valid[1].double(),c.valid[2].double()
    W=net.head.weight.detach().double();b=net.head.bias.detach().double();L=2 if c.reg else 1
    T=c.T;theta=torch.zeros(T,129,dtype=torch.float64);thn=torch.zeros(T,129,dtype=torch.float64)
    Hs=torch.zeros(T,129,129,dtype=torch.float64);covs=torch.zeros(T,129,129,dtype=torch.float64);alphas=[];report=[]
    for j in range(T):
        r=j*L;th=torch.cat([W[r],b[r:r+1]]);theta[j]=th;k=mva[:,j].bool()
        if c.reg:
            tn=torch.cat([W[r+1],b[r+1:r+2]]);thn[j]=tn
            noise_tr=F.softplus(Ptr@tn).clamp_min(1e-6);noise_va=F.softplus(Pva@tn).clamp_min(1e-6)
            H=hessian_reg(Ptr,noise_tr,mtr[:,j]);mu=Pva@th
            det=nll_reg(yva[k,j],mu[k],noise_va[k])
            grid=[]
            for a in ALPHAS:
                grid.append((a,nll_reg(yva[k,j],mu[k],(noise_va+var_of(Pva,cov_from(H,a)))[k])))
        else:
            H=hessian_cls(Ptr,th,mtr[:,j]);z=Pva@th
            det=nll_cls(yva[k,j],torch.sigmoid(z)[k])
            grid=[(a,nll_cls(yva[k,j],probit(z,var_of(Pva,cov_from(H,a)))[k])) for a in ALPHAS]
        alpha,score=min(grid,key=lambda g:g[1]);Hs[j]=H;covs[j]=cov_from(H,alpha);alphas.append(alpha)
        report.append({'task':j,'observed_valid':int(k.sum()),'alpha':alpha,'validation_nll_laplace':score,'validation_nll_deterministic':det,'grid':grid})
    arrays={'theta':theta.numpy(),'theta_noise':thn.numpy(),'hessian':Hs.numpy(),'covariance':covs.numpy(),'alpha':np.asarray(alphas)}
    return arrays,{'split':c.split,'dataset':c.d,'seed':seed,'tasks':T,'fit_seconds':time.perf_counter()-tick,'per_task':report,
                   'mean_validation_nll_laplace':float(np.mean([r['validation_nll_laplace'] for r in report])),
                   'mean_validation_nll_deterministic':float(np.mean([r['validation_nll_deterministic'] for r in report]))}


import csv
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
LAMBDAS=[0.,.01,.1,1.,10.,100.];VARIANTS=['Laplace constant','Novelty only','Novelty Laplace'];_FP={}
def distances(split,d,parts):
    """d(x) = 1 - max Tanimoto (Morgan r2, 2048 bits, no chirality) to the TRAINING molecules of this split. Unparsable query SMILES get d=1."""
    if d not in _FP:
        sm=[r['smiles'] for part in ['train','valid','test'] for r in csv.DictReader((V4/'data'/d/f'{part}.csv').open())]
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
        _FP[d]=[(gen.GetFingerprint(m) if (m:=Chem.MolFromSmiles(x)) is not None else None) for x in sm]
    ii=t.split_index(d,split);ref=[_FP[d][i] for i in ii['train'] if _FP[d][i] is not None]
    return {p:np.array([1.0 if _FP[d][i] is None else 1-max(DataStructs.BulkTanimotoSimilarity(_FP[d][i],ref)) for i in ii[p]]) for p in parts}

def part_outputs(c,net,arrays,part):
    """Logits or standardised means mu, Laplace variance v and predicted noise (regression) for the rows of one part."""
    P=phi(feats(net,getattr(c,part)[0]));th=torch.from_numpy(arrays['theta']);cov=torch.from_numpy(arrays['covariance'])
    mu=torch.stack([P@th[j] for j in range(c.T)],1);v=torch.stack([var_of(P,cov[j]) for j in range(c.T)],1);noise=None
    if c.reg:tn=torch.from_numpy(arrays['theta_noise']);noise=torch.stack([F.softplus(P@tn[j]) for j in range(c.T)],1)
    return mu,v,noise

def compose(c,mu,v,noise,dist,variant,lam):
    """Plain Laplace: variant None. Variants as expanded_v2: initial variance (Laplace, or zero for novelty-only) + lam * (1 or d^2)."""
    if variant is None:var=v
    else:
        d2=(torch.as_tensor(dist)**2)[:,None].expand_as(v);base=torch.zeros_like(v) if variant=='Novelty only' else v
        var=base+lam*(torch.ones_like(v) if variant=='Laplace constant' else d2)
    return (probit(mu,var),None) if not c.reg else (mu,noise+var)

def val_nll(c,out,j):
    k=c.valid[2][:,j].bool();y=c.valid[1].double()[k,j];m,v=out
    return nll_cls(y,m[k,j]) if not c.reg else nll_reg(y,m[k,j],v[k,j].clamp_min(1e-6))

def fit_novelty(c,seed,arrays):
    net,_=t.fit(c,'det',seed);mu,v,noise=part_outputs(c,net,arrays,'valid');dist=distances(c.split,c.d,['valid'])['valid'];res={}
    for var in VARIANTS:
        rows=[]
        for j in range(c.T):
            grid=[(l,val_nll(c,compose(c,mu,v,noise,dist,var,l),j)) for l in LAMBDAS];best=min(grid,key=lambda g:g[1])
            rows.append({'task':j,'lambda':best[0],'validation_nll':best[1],'grid':grid})
        res[var]=rows
    return {'valid_distance':{'n':int(len(dist)),'novel_d_gt_0.6':int((dist>0.6).sum()),'mean':float(dist.mean())},'variants':res}

def fit_novelty_all(splits,datasets):
    for split in splits:
        for d in datasets:
            marker=V6/'frozen'/f'NOV_{split}_{d}.json'
            if marker.exists():continue
            seeds=t.seeds(split)[0];c=None;edge=[]
            for s in seeds:
                f=V6/'results'/split/d/f'seed-{s}'
                if not (f/'novelty.json').exists():
                    c=c or t.Ctx(split,d)
                    with np.load(f/'laplace_posterior.npz') as z:arrays={k:z[k] for k in z.files}
                    e.save_json(f/'novelty.json',fit_novelty(c,s,arrays))
                rep=json.loads((f/'novelty.json').read_text())
                for var,rows in rep['variants'].items():
                    for r in rows:
                        if r['lambda'] in (LAMBDAS[-1],):edge.append({'seed':s,'variant':var,'task':r['task'],'lambda':r['lambda']})
                print(json.dumps({'split':split,'dataset':d,'seed':s,'novel_valid':rep['valid_distance']['novel_d_gt_0.6'],'n_valid':rep['valid_distance']['n'],
                      **{v:[round(r['lambda'],3) for r in rows][:3] for v,rows in rep['variants'].items()}}),flush=True)
            e.save_json(marker,{'at_utc':e.now(),'split':split,'dataset':d,'seeds':seeds,'protocol_sha256':e.digest(V6/'PROTOCOL.md'),'code_sha256':e.digest(__file__),
                                'lambda_at_upper_edge':edge,'test_unopened_by_laplace':True});print('FROZEN NOV',split,d,flush=True)

from types import SimpleNamespace
import tierb_eval as ev
def _sub(c,sel):return SimpleNamespace(T=c.T,reg=c.reg,raw_test=c.raw_test[sel],mask_test=c.mask_test[sel],ids={'test':c.ids['test'][sel]})
def _convert(c,m,v):
    if c.reg:return (m.numpy()*c.scale+c.shift,v.numpy()*c.scale**2)
    return m.numpy(),None

def evaluate_all(splits,datasets):
    """Test predictions and metrics, only behind BOTH FROZEN markers (primary alpha, novelty lambda) with matching protocol hash."""
    psha=e.digest(V6/'PROTOCOL.md')
    for split in splits:
        for d in datasets:
            m1=V6/'frozen'/f'DNN_{split}_{d}.json';m2=V6/'frozen'/f'NOV_{split}_{d}.json';res=V6/'results'/split/d
            if not (m1.exists() and m2.exists()) or (res/'per_seed_metrics.json').exists():continue
            assert json.loads(m1.read_text())['protocol_sha256']==psha and json.loads(m2.read_text())['protocol_sha256']==psha
            c=t.Ctx(split,d);dist=distances(split,d,['test'])['test'];assert len(dist)==len(c.raw_test);rows=[];novel=dist>0.6
            for s in t.seeds(split)[0]:
                f=res/f'seed-{s}';net,_=t.fit(c,'det',s)
                with np.load(f/'laplace_posterior.npz') as z:arrays={k:z[k] for k in z.files}
                nov=json.loads((f/'novelty.json').read_text())['variants'];mu,v,noise=part_outputs(c,net,arrays,'test')
                preds={'Laplace':compose(c,mu,v,noise,dist,None,0.)}
                for var in VARIANTS:preds[var]=compose(c,mu,v,noise,dist,var,torch.tensor([r['lambda'] for r in nov[var]],dtype=torch.float64))
                for name,(pm,pv) in preds.items():
                    mean,variance=_convert(c,pm,pv);assert np.isfinite(mean).all() and (variance is None or (np.isfinite(variance).all() and (variance>0).all()))
                    for grp,sel in [('all',np.ones(len(dist),bool)),('novel',novel),('familiar',~novel)]:
                        n=int(sel.sum());row={'backbone':'DNN','split':split,'dataset':d,'method':name,'seed':s,'subgroup':grp,'n':n}
                        if grp=='all':row.update(ev.macro(c,mean,variance))
                        elif n>=20:
                            try:row.update(ev.macro(_sub(c,sel),mean[sel],None if variance is None else variance[sel]))
                            except Exception as ex:row['not_estimable']=str(ex)[:80]
                        else:row['not_estimable']='fewer than 20 molecules'
                        rows.append(row)
                    arrays_out=dict(mean=mean,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
                    if variance is not None:arrays_out['variance']=variance
                    np.savez_compressed(res/f'{name.replace(" ","_")}-{s}.npz',**arrays_out)
            e.save_json(res/'per_seed_metrics.json',rows);print('EVALUATED',split,d,len(rows),'novel test',int(novel.sum()),'of',len(novel),flush=True)

def fit_all(splits,datasets):
    """Fit alpha per task on validation data for every split/dataset/seed, then write the FROZEN marker. No test data used."""
    for split in splits:
        for d in datasets:
            marker=V6/'frozen'/f'DNN_{split}_{d}.json'
            if marker.exists():continue
            seeds=t.seeds(split)[0];c=None;boundary=[];counts=0
            for s in seeds:
                f=V6/'results'/split/d/f'seed-{s}'
                if not (f/'fit.json').exists():
                    c=c or t.Ctx(split,d);arrays,rep=fit_group(c,s);f.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(f/'laplace_posterior.npz',**arrays);e.save_json(f/'fit.json',rep)
                rep=json.loads((f/'fit.json').read_text())
                for r in rep['per_task']:
                    counts+=1
                    if r['alpha'] in (ALPHAS[0],ALPHAS[-1]):boundary.append({'seed':s,'task':r['task'],'alpha':r['alpha']})
                print(json.dumps({'split':split,'dataset':d,'seed':s,'tasks':rep['tasks'],'valid_nll_laplace':round(rep['mean_validation_nll_laplace'],4),
                                  'valid_nll_det':round(rep['mean_validation_nll_deterministic'],4),'fit_s':round(rep['fit_seconds'],2)}),flush=True)
            marker.parent.mkdir(exist_ok=True)
            e.save_json(marker,{'at_utc':e.now(),'backbone':'DNN','split':split,'dataset':d,'seeds':seeds,'protocol_sha256':e.digest(V6/'PROTOCOL.md'),
                                'code_sha256':e.digest(__file__),'task_fits':counts,'alpha_at_grid_boundary':boundary,'test_unopened_by_laplace':True})
            print('FROZEN',split,d,'boundary hits',len(boundary),'of',counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['checks','smoke','fit','novelty','evaluate']);p.add_argument('--splits',nargs='*',default=['scaffold']);p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args()
    ck=checks();print('CHECKS',json.dumps(ck))
    if a.stage=='fit':fit_all(t.SPLITS if a.splits==['scaffold'] and not os.environ.get('V6_SCAFFOLD_ONLY') else a.splits,a.datasets)
    if a.stage=='novelty':fit_novelty_all(t.SPLITS,a.datasets)
    if a.stage=='evaluate':evaluate_all(t.SPLITS,a.datasets)
    if a.stage=='smoke':
        for split in a.splits:
            for d in a.datasets:
                c=t.Ctx(split,d);seed=t.seeds(split)[0][0];arrays,rep=fit_group(c,seed)
                print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in rep.items() if k!='per_task'}))
                for r in rep['per_task'][:3]:print('  task',r['task'],'alpha',r['alpha'],'valid NLL laplace',round(r['validation_nll_laplace'],5),'deterministic',round(r['validation_nll_deterministic'],5))
                print('  grid',[(f'{g[0]:g}',round(g[1],4)) for g in rep['per_task'][0]['grid']])
