"""Pinned MUBen BBP and SGLD heads on ESOL; paper/source sampling ablation."""
from pathlib import Path
import sys,json,time,math,copy
ROOT=Path(__file__).resolve().parent;V2=ROOT.parent/'expanded_v2';sys.path.insert(0,str(V2))
import run_expansion as e
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
bbp=e.module_from('source_bbp',ROOT/'source/muben/uncertainty/bbp.py')
SGLD=e.module_from('source_sgld',ROOT/'source/muben/uncertainty/sgld.py').SGLDOptimizer

class BayesianModel(e.Model):
    def __init__(self):
        super().__init__(True);self.head=bbp.BBPOutputLayer(128,2,bbp_prior_sigma=.1)
    def forward(self,x):
        out,self.kl=self.head(self.features(x));return out

def checks():
    e.seed_all(9123);mu=torch.randn(13,dtype=torch.float64);sd=torch.rand(13,dtype=torch.float64)+.01
    actual=bbp.kld_cost(0,.1,mu,sd);expected=torch.distributions.kl_divergence(torch.distributions.Normal(mu,sd),torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(sd)*.1)).sum();torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    p=nn.Parameter(torch.tensor([.3,-.1],dtype=torch.float64));old=p.detach().clone();grad=torch.tensor([.2,.5],dtype=torch.float64);p.grad=grad.clone();SGLD([p],lr=.0002,norm_sigma=.1,addnoise=False).step();torch.testing.assert_close(p,old-.0001*(grad+100*old),rtol=1e-12,atol=1e-12)
    p=nn.Parameter(torch.zeros(100000));p.grad=torch.zeros_like(p);SGLD([p],lr=.0002).step();variance=float(p.detach().var());assert abs(variance/.0002-1)<.02
    e.save_json(ROOT/'BAYESIAN_CHECKS.json',{'bbp_kl_matches_torch':True,'sgld_drift_matches_algebra':True,'sgld_noise_variance':variance,'expected_noise_variance':.0002})

def fit(kind,seed):
    folder=ROOT/'bayesian_models'/kind/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return
    x,y,_,_=e.data('esol','train');vx,vy,_,_=e.data('esol','valid');shift=float(y.double().mean());scale=float(y.double().std(unbiased=False));y=(y-shift)/scale;vy=(vy-shift)/scale
    e.seed_all(seed);m=BayesianModel() if kind=='BBP' else e.Model(True)
    opt=torch.optim.AdamW(m.parameters() if kind=='BBP' else m.features.parameters(),lr=.0002,weight_decay=.01)
    sampler=None if kind=='BBP' else SGLD(m.head.parameters(),lr=.0002,norm_sigma=.1)
    gen=torch.Generator().manual_seed(seed);loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=256,shuffle=True,generator=gen)
    def train_epoch():
        m.train();total=0.
        for bx,by in loader:
            m.zero_grad(set_to_none=True);out=m(bx);loss=e.loss(out,by,True)
            if kind=='BBP':loss=loss+m.kl/(len(loader)*len(by))
            assert torch.isfinite(loss);loss.backward();opt.step()
            if sampler:sampler.step()
            total+=float(loss.detach())*len(by)
        return total/len(y)
    tick=time.perf_counter();hist=[];best=-float('inf');stale=0;start=0;before=0
    if (folder/'resume.pt').exists():
        r=torch.load(folder/'resume.pt',weights_only=True);assert r['source_sha256']==e.digest(__file__);m.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);gen.set_state(r['loader_rng']);torch.set_rng_state(r['rng']);hist,best,stale,start,before=r['history'],r['best'],r['stale'],r['epoch'],r['seconds']
        if sampler:sampler.load_state_dict(r['sampler'])
    for ep in range(start,400):
        if stale>50:break
        l=train_epoch();vo=e.predict(m,vx);score=-float(((vo[:,0]-vy)**2).mean().sqrt())
        if score>=best:best=score;stale=0;e.atomic_torch(m.state_dict(),folder/'best.pt')
        else:stale+=1
        hist.append({'epoch':ep+1,'loss':l,'validation_selection':score})
        if (ep+1)%10==0 or stale>50 or ep==399:e.atomic_torch(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict() if sampler else {},rng=torch.get_rng_state(),loader_rng=gen.get_state(),history=hist,best=best,stale=stale,epoch=ep+1,seconds=before+time.perf_counter()-tick,source_sha256=e.digest(__file__)),folder/'resume.pt')
    m.load_state_dict(torch.load(folder/'best.pt',weights_only=True))
    # Replay one stochastic validation draw after reload with identical RNG.
    rng=torch.get_rng_state();a=e.predict(m,vx);m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));torch.set_rng_state(rng);torch.testing.assert_close(a,e.predict(m,vx),rtol=0,atol=0)
    if sampler:
        # Preserve optimizer state at end of training, as upstream does. A single
        # trajectory supplies first30 snapshots and every-second of60 snapshots.
        snap=folder/'snapshots';snap.mkdir(exist_ok=True)
        for ep in range(60):
            l=train_epoch();e.atomic_torch(m.state_dict(),snap/f'epoch-{ep+1:02d}.pt')
        e.atomic_torch(dict(model=m.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict(),rng=torch.get_rng_state(),loader_rng=gen.get_state()),folder/'sampling_final.pt')
    e.save_json(folder/'history.json',hist);e.save_json(folder/'summary.json',{'kind':kind,'seed':seed,'epochs':len(hist),'best_epoch':max(r['epoch'] for r in hist if r['validation_selection']==best),'target_mean':shift,'target_std':scale,'training_and_sampling_seconds':before+time.perf_counter()-tick,'reload_verified':True,'source_sha256':e.digest(__file__)})
    print('Fitted',kind,seed,flush=True)

def evaluate():
    tx,ty,_,ids=e.data('esol','test');rows=[];out=ROOT/'bayesian_results';out.mkdir(exist_ok=True)
    for kind in ['BBP','SGLD']:
        for seed in [0,100,200]:
            folder=ROOT/'bayesian_models'/kind/f'seed-{seed}';meta=json.loads((folder/'summary.json').read_text());e.seed_all(seed+50000)
            m=BayesianModel() if kind=='BBP' else e.Model(True);draws={}
            if kind=='BBP':
                m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));draws['BBP']=[e.predict(m,tx) for _ in range(30)]
            else:
                all_draws=[]
                for ep in range(1,61):m.load_state_dict(torch.load(folder/'snapshots'/f'epoch-{ep:02d}.pt',weights_only=True));all_draws.append(e.predict(m,tx))
                draws['SGLD paper interval1']=all_draws[:30];draws['SGLD source interval2']=all_draws[1::2]
            for name,oo in draws.items():
                raw_mu=torch.stack([o[:,0] for o in oo]).numpy();raw_var=torch.stack([e.noise(o) for o in oo]).numpy();mu=raw_mu.mean(0)*meta['target_std']+meta['target_mean'];var=raw_var.mean(0)*meta['target_std']**2
                assert np.isfinite(mu).all() and np.isfinite(var).all() and (var>0).all();met,risk=e.reg_metrics(ty.numpy(),mu,var,ids);rows.append({'dataset':'esol','seed':seed,'method':name,**met})
                np.savez_compressed(out/f'{seed}-{name.replace(" ","_")}.npz',labels=ty.numpy(),mean=mu,variance=var,ids=ids,risk=risk,raw_standardized_mean_draws=raw_mu,raw_standardized_variance_draws=raw_var)
    e.save_json(out/'per_seed_metrics.json',rows)

if __name__=='__main__':
    checks()
    for kind in ['BBP','SGLD']:
        for seed in [0,100,200]:fit(kind,seed)
    e.save_json(ROOT/'BAYESIAN_FROZEN.json',{'at_utc':e.now(),'source_sha256':e.digest(__file__),'test_unopened':True})
    evaluate()
