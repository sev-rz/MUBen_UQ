"""Laplace on the Tier-B ChemBERTa nets (protocol: chemberta/PROTOCOL.md, sha256 in PROTOCOL.sha256). Stage-1 maths (tierb_laplace.py) reused unchanged;
only feature extraction (pooler output of the frozen backbone) and paths differ. Fit and novelty stages use TRAIN and VALID tokens only; test tokens are touched only in evaluate, behind both FROZEN markers."""
import os, sys, json, time, argparse
from pathlib import Path
CB=Path(__file__).resolve().parent;V6=CB.parent;WORK=V6.parent;V4=WORK/'expanded_v4'
sys.path.insert(0,str(V4));sys.path.insert(0,str(V6))
import tierb_chemberta as cb          # first: restores GPU visibility that tierb_dnn hides
import tierb_laplace as L
import numpy as np, torch
from torch.nn import functional as F
t,e=cb.t,cb.e
OUT=CB;ALPHAS=L.ALPHAS;LAMBDAS=L.LAMBDAS;VARIANTS=L.VARIANTS

def feats(net,tokens):
    net.eval();out=[]
    with torch.no_grad():
        for s in range(0,len(tokens),cb.BATCH):
            b=cb.batch(tokens,range(s,min(s+cb.BATCH,len(tokens))));out.append(net.backbone(**b).pooler_output.double().cpu())
    return torch.cat(out)

def fit_group(c,seed):
    tick=time.perf_counter();net,_=cb.fit(c,'det',seed);net.eval()
    Ptr=L.phi(feats(net,c.tokens['train']));Pva=L.phi(feats(net,c.tokens['valid']));D=Ptr.shape[1];T=c.T;Lh=2 if c.reg else 1
    ytr,mtr=c.train[1].double(),c.train[2].double();yva,mva=c.valid[1].double(),c.valid[2].double()
    W=net.head.weight.detach().double().cpu();b=net.head.bias.detach().double().cpu()
    theta=torch.zeros(T,D,dtype=torch.float64);thn=torch.zeros(T,D,dtype=torch.float64);covs=torch.zeros(T,D,D,dtype=torch.float64);alphas=[];report=[]
    ref=cb.predict(net,c.tokens['valid'][:64]).double();worst=0.   # features must reproduce the saved net's outputs
    for j in range(T):
        r=j*Lh;th=torch.cat([W[r],b[r:r+1]]);theta[j]=th;k=mva[:,j].bool()
        mine=(Pva[:64]@th);worst=max(worst,float((mine-(ref[:,j,0] if c.reg else ref[:,j])).abs().max()))
        if c.reg:
            tn=torch.cat([W[r+1],b[r+1:r+2]]);thn[j]=tn
            noise_tr=F.softplus(Ptr@tn).clamp_min(1e-6);noise_va=F.softplus(Pva@tn).clamp_min(1e-6)
            H=L.hessian_reg(Ptr,noise_tr,mtr[:,j]);mu=Pva@th;det=L.nll_reg(yva[k,j],mu[k],noise_va[k])
            grid=[(a,L.nll_reg(yva[k,j],mu[k],(noise_va+L.var_of(Pva,L.cov_from(H,a)))[k])) for a in ALPHAS]
        else:
            H=L.hessian_cls(Ptr,th,mtr[:,j]);z=Pva@th;det=L.nll_cls(yva[k,j],torch.sigmoid(z)[k])
            grid=[(a,L.nll_cls(yva[k,j],L.probit(z,L.var_of(Pva,L.cov_from(H,a)))[k])) for a in ALPHAS]
        alpha,score=min(grid,key=lambda g:g[1]);covs[j]=L.cov_from(H,alpha);alphas.append(alpha)
        report.append({'task':j,'observed_valid':int(k.sum()),'alpha':alpha,'validation_nll_laplace':score,'validation_nll_deterministic':det,'grid':grid})
    assert worst<1e-2,f'feature/head outputs differ from the saved net by {worst}'
    arrays={'theta':theta.numpy(),'theta_noise':thn.numpy(),'covariance':covs.numpy(),'alpha':np.asarray(alphas)}
    return arrays,{'split':c.split,'dataset':c.d,'seed':seed,'tasks':T,'dim':D,'max_abs_diff_vs_saved_net':worst,'fit_seconds':time.perf_counter()-tick,'per_task':report,
                   'mean_validation_nll_laplace':float(np.mean([r['validation_nll_laplace'] for r in report])),'mean_validation_nll_deterministic':float(np.mean([r['validation_nll_deterministic'] for r in report]))}

def part_outputs(c,net,arrays,part):
    P=L.phi(feats(net,c.tokens[part]));th=torch.from_numpy(arrays['theta']);cov=torch.from_numpy(arrays['covariance'])
    mu=torch.stack([P@th[j] for j in range(c.T)],1);v=torch.stack([L.var_of(P,cov[j]) for j in range(c.T)],1);noise=None
    if c.reg:tn=torch.from_numpy(arrays['theta_noise']);noise=torch.stack([F.softplus(P@tn[j]) for j in range(c.T)],1)
    return mu,v,noise

def fit_all():
    for split in t.SPLITS:
        for d in t.DATASETS:
            mk=OUT/'frozen'/f'ChemBERTa_{split}_{d}.json'
            if mk.exists():continue
            c=None;edge=[];n=0
            for s in t.seeds(split)[0]:
                f=OUT/'results'/split/d/f'seed-{s}'
                if not (f/'fit.json').exists():
                    c=c or cb.CCtx(split,d);arr,rep=fit_group(c,s);f.mkdir(parents=True,exist_ok=True);np.savez_compressed(f/'laplace_posterior.npz',**arr);e.save_json(f/'fit.json',rep)
                rep=json.loads((f/'fit.json').read_text())
                for r in rep['per_task']:
                    n+=1
                    if r['alpha'] in (ALPHAS[0],ALPHAS[-1]):edge.append({'seed':s,'task':r['task'],'alpha':r['alpha']})
                print(json.dumps({'split':split,'dataset':d,'seed':s,'tasks':rep['tasks'],'dim':rep['dim'],'valid_nll_laplace':round(rep['mean_validation_nll_laplace'],4),
                                  'valid_nll_det':round(rep['mean_validation_nll_deterministic'],4),'maxdiff':round(rep['max_abs_diff_vs_saved_net'],6),'fit_s':round(rep['fit_seconds'],1)}),flush=True)
            mk.parent.mkdir(exist_ok=True)
            e.save_json(mk,{'at_utc':e.now(),'backbone':'ChemBERTa','split':split,'dataset':d,'seeds':t.seeds(split)[0],'protocol_sha256':e.digest(CB/'PROTOCOL.md'),'code_sha256':e.digest(__file__),
                            'task_fits':n,'alpha_at_grid_boundary':edge,'test_unopened_by_laplace':True});print('FROZEN',split,d,'edge hits',len(edge),'of',n,flush=True)

def fit_novelty(c,seed,arrays):
    net,_=cb.fit(c,'det',seed);mu,v,noise=part_outputs(c,net,arrays,'valid');dist=L.distances(c.split,c.d,['valid'])['valid'];res={}
    for var in VARIANTS:
        rows=[]
        for j in range(c.T):
            grid=[(l_,L.val_nll(c,L.compose(c,mu,v,noise,dist,var,l_),j)) for l_ in LAMBDAS];best=min(grid,key=lambda g:g[1]);rows.append({'task':j,'lambda':best[0],'validation_nll':best[1],'grid':grid})
        res[var]=rows
    return {'valid_distance':{'n':int(len(dist)),'novel_d_gt_0.6':int((dist>0.6).sum()),'mean':float(dist.mean())},'variants':res}

def novelty_all():
    for split in t.SPLITS:
        for d in t.DATASETS:
            mk=OUT/'frozen'/f'NOV_ChemBERTa_{split}_{d}.json'
            if mk.exists():continue
            c=None;edge=[]
            for s in t.seeds(split)[0]:
                f=OUT/'results'/split/d/f'seed-{s}'
                if not (f/'novelty.json').exists():
                    c=c or cb.CCtx(split,d)
                    with np.load(f/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
                    e.save_json(f/'novelty.json',fit_novelty(c,s,arr))
                for var,rows in json.loads((f/'novelty.json').read_text())['variants'].items():
                    edge+=[{'seed':s,'variant':var,'task':r['task'],'lambda':r['lambda']} for r in rows if r['lambda']==LAMBDAS[-1]]
            e.save_json(mk,{'at_utc':e.now(),'split':split,'dataset':d,'protocol_sha256':e.digest(CB/'PROTOCOL.md'),'code_sha256':e.digest(__file__),'lambda_at_upper_edge':edge,'test_unopened_by_laplace':True});print('FROZEN NOV',split,d,flush=True)

def evaluate_all():
    psha=e.digest(CB/'PROTOCOL.md')
    for split in t.SPLITS:
        for d in t.DATASETS:
            m1=OUT/'frozen'/f'ChemBERTa_{split}_{d}.json';m2=OUT/'frozen'/f'NOV_ChemBERTa_{split}_{d}.json';res=OUT/'results'/split/d
            if not (m1.exists() and m2.exists()) or (res/'per_seed_metrics.json').exists():continue
            assert json.loads(m1.read_text())['protocol_sha256']==psha and json.loads(m2.read_text())['protocol_sha256']==psha
            c=cb.CCtx(split,d);dist=L.distances(split,d,['test'])['test'];assert len(dist)==len(c.raw_test);rows=[];novel=dist>0.6
            for s in t.seeds(split)[0]:
                f=res/f'seed-{s}';net,_=cb.fit(c,'det',s)
                with np.load(f/'laplace_posterior.npz') as z:arr={k:z[k] for k in z.files}
                nov=json.loads((f/'novelty.json').read_text())['variants'];mu,v,noise=part_outputs(c,net,arr,'test');preds={'Laplace':L.compose(c,mu,v,noise,dist,None,0.)}
                for var in VARIANTS:preds[var]=L.compose(c,mu,v,noise,dist,var,torch.tensor([r['lambda'] for r in nov[var]],dtype=torch.float64))
                for name,(pm,pv) in preds.items():
                    mean,variance=L._convert(c,pm,pv);assert np.isfinite(mean).all() and (variance is None or (np.isfinite(variance).all() and (variance>0).all()))
                    for grp,sel in [('all',np.ones(len(dist),bool)),('novel',novel),('familiar',~novel)]:
                        n=int(sel.sum());row={'backbone':'ChemBERTa','split':split,'dataset':d,'method':name,'seed':s,'subgroup':grp,'n':n}
                        if grp=='all':row.update(L.ev.macro(c,mean,variance))
                        elif n>=20:
                            try:row.update(L.ev.macro(L._sub(c,sel),mean[sel],None if variance is None else variance[sel]))
                            except Exception as ex:row['not_estimable']=str(ex)[:80]
                        else:row['not_estimable']='fewer than 20 molecules'
                        rows.append(row)
                    out={'mean':mean,'labels':c.raw_test,'masks':c.mask_test,'ids':c.ids['test']}
                    if variance is not None:out['variance']=variance
                    np.savez_compressed(res/f'{name.replace(" ","_")}-{s}.npz',**out)
            e.save_json(res/'per_seed_metrics.json',rows);print('EVALUATED',split,d,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','all']);p.add_argument('--datasets',nargs='*',default=['freesolv','bace']);a=p.parse_args()
    print('CHECKS',json.dumps(L.checks()),'| device',cb.DEV)
    if a.stage=='smoke':
        for d in a.datasets:
            c=cb.CCtx('scaffold',d);arr,rep=fit_group(c,0)
            print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in rep.items() if k!='per_task'}));r=rep['per_task'][0];print('  task 0 alpha',r['alpha'],'valid NLL laplace',round(r['validation_nll_laplace'],4),'det',round(r['validation_nll_deterministic'],4))
    else:fit_all();novelty_all();evaluate_all();print('STAGE2 DONE')
