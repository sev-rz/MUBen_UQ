"""Shared setup for round 2 (PROTOCOL.md): backbone modules (unchanged stage code), recalibration baseline, Hessian capture, evidence alpha, per-molecule NLL, R10 verdict."""
import os,sys,json,time,math,itertools
from pathlib import Path
from types import SimpleNamespace
import numpy as np
D=Path(__file__).resolve().parent;W=D.parent
V4=W/'expanded_v4';V6=W/'expanded_v6_laplace';V8=W/'expanded_v8_gin';V10=W/'expanded_v10_unimol';V14=W/'expanded_v14_laplace10'
DS=['bace','bbbp','clintox','tox21','esol','freesolv','lipo','qm7'];REG={'esol','freesolv','lipo','qm7'}
def setup(bb):
    sys.path.insert(0,str(V4));sys.path.insert(0,str(V6))
    if bb=='GIN':os.environ['TIERB_OUT']=str(V8);sys.path.insert(0,str(V8));sys.path.insert(0,str(V8/'laplace'))
    if bb=='Uni-Mol':os.environ['TIERB_OUT']=str(V10);sys.path.insert(0,str(V10));sys.path.insert(0,str(V10/'laplace'))
    if bb=='ChemBERTa':sys.path.insert(0,str(V6/'chemberta'))
    import torch
    if bb=='DNN':
        import tierb_laplace as L;t=L.t;mk=t.Ctx;fg=L.fit_group;po=L.part_outputs;folder=t.folder
        import tierb_eval as ev;det=lambda c,net:ev.convert(c,t.e.predict(net,c.test[0]));newnet=lambda c,s:t.fit(c,'det',s)[0]
    elif bb=='ChemBERTa':
        import chemberta_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.CCtx;fg=M.fit_group;po=M.part_outputs;folder=cb.folder
        import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.tokens['test']));newnet=lambda c,s:cb.fit(c,'det',s)[0]
    elif bb=='GIN':
        import gin_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.GCtx;fg=M.fit_group;po=M.part_outputs;folder=cb.folder
        import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.predict(net,c.graphs['test']));newnet=lambda c,s:cb.fit(c,'det',s)[0]
    else:
        import unimol_laplace as M;import tierb_laplace as L;cb=M.cb;mk=cb.UCtx;fg=M.fit_group;po=M.part_outputs;folder=cb.folder
        import tierb_eval as ev;det=lambda c,net:ev.convert(c,cb.agg(cb.predict_raw(net,c,'test'),c.reg));newnet=lambda c,s:cb.fit(c,'det',s)[0]
    seeds=list(range(10))+([] if bb=='Uni-Mol' else [100,200])
    return SimpleNamespace(L=L,mk=mk,fit_group=fg,po=po,folder=folder,det=det,newnet=newnet,ev=ev,seeds=seeds,torch=torch)
def scale_params(S,c,mu,noise):
    torch=S.torch;L=S.L;TG=torch.logspace(np.log10(0.05),np.log10(20.),81,dtype=torch.float64)
    y=c.valid[1].double();m=c.valid[2].double();out=[]
    for j in range(c.T):
        k=m[:,j].bool()
        if c.reg:out.append(float(((y[k,j]-mu[k,j])**2/noise[k,j].clamp_min(1e-6)).mean()))
        else:
            z=mu[k,j];yy=y[k,j];sc=[L.nll_cls(yy,torch.sigmoid(z/T)) for T in TG];out.append(float(TG[int(np.argmin(sc))]))
    return torch.tensor(out,dtype=torch.float64)
def scaled(S,c,mu,noise,par):
    return (mu,noise*par[None,:]) if c.reg else (S.torch.sigmoid(mu/par[None,:]),None)
def capture(L):
    """Wrap L.cov_from so that the data Hessian of every task is recorded (in task order); returns (CAP, original)."""
    CAP=[];last=[None];orig=L.cov_from
    def cap(H,alpha):
        if last[0] is not H:CAP.append(H.detach().clone());last[0]=H
        return orig(H,alpha)
    L.cov_from=cap;return CAP,orig
EGRID=[10.**(-8+12*i/120) for i in range(121)]
def evidence_alpha(torch,H,theta):
    """argmax over EGRID of E(a) = D/2 log a - a/2 ||theta||^2 - 1/2 sum_k log(lambda_k + a); H is the data Hessian (no prior)."""
    lam=torch.linalg.eigvalsh(H).clamp_min(0);a=torch.tensor(EGRID,dtype=H.dtype);th2=float((theta**2).sum());Dm=len(theta)
    E=0.5*Dm*torch.log(a)-0.5*a*th2-0.5*torch.log(lam[None,:]+a[:,None]).sum(1)
    return float(a[int(torch.argmax(E))])
def per_mol_nll(mean,var,y,m,reg):
    if reg:return 0.5*(np.log(var)+(y-mean)**2/var),m
    with np.errstate(divide='ignore'):lp=np.maximum(np.log(mean),-100.0);l1=np.maximum(np.log(1-mean),-100.0)
    return -(y*lp+(1-y)*l1),m
def scored_tasks(y,m,reg):
    if reg:return np.ones(y.shape[1],bool)
    return np.array([0<y[m[:,j],j].mean()<1 for j in range(y.shape[1])])
def macro_nll(nll,m,ok,idx=None):
    n0=np.where(m,nll,0.0)
    if idx is None:return float(np.mean([n0[:,j].sum()/m[:,j].sum() for j in range(nll.shape[1]) if ok[j]]))
    Ssum=n0[idx].sum(1);N=m[idx].sum(1);return np.mean([Ssum[:,j]/N[:,j] for j in range(nll.shape[1]) if ok[j]],axis=0)
def verdict(x,pos='better',neg='worse'):
    from scipy import stats
    x=np.asarray(x,float);n=len(x);mn=x.mean();se=x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;t=stats.t.ppf(0.975,n-1);lo,hi=mn-t*se,mn+t*se
    if (x<0).mean()>=0.9 and hi<0:v=pos
    elif (x>0).mean()>=0.9 and lo>0:v=neg
    else:v='unclear'
    return v,mn,lo,hi,(x<0).mean(),(mn/se if se>0 else 0.)
def signflip(z):
    z=np.asarray(z,float);obs=z.sum();return float(np.mean([np.dot(s,z)<=obs+1e-12 for s in itertools.product([1,-1],repeat=len(z))]))
