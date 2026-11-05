"""CPU MUBen expansion. Run --stage fit, then --stage evaluate.

All artifacts are relative to this script. Requires the preserved parent
train_baseline.py/evaluate_uncertainty.py and pinned upstream source. See
PROTOCOL.md for model equations, test isolation, provenance and deviations.
"""
import os
os.environ['MPLBACKEND']='Agg'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, ast, csv, json, time, copy, hashlib, random, argparse, importlib.util, resource
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from train_baseline import save_json, digest
from evaluate_uncertainty import fit_laplace, features, phi, metrics as old_metrics, scientific_checks
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from scipy.special import expit, ndtr
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from descriptastorus.descriptors import rdNormalizedDescriptors
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.functional.classification import binary_calibration_error

DATASETS=['bace','bbbp','esol']; GROUPS=[list(range(s,s+5)) for s in [0,100,200]]
LAMBDA=[0.,.01,.1,1.,10.,100.]; ALPHA=[10.**a for a in range(-4,5)]
CONFIG={'datasets':DATASETS,'groups':GROUPS,'lambda_grid':LAMBDA,'alpha_grid':ALPHA,
        'epochs':400,'patience_stop_strictly_greater':50,'mc_samples':30,'swag_epochs':20,
        'fingerprints':{'radius':2,'fpSize':2048,'includeChirality':False},
        'primary_architecture':'8x128 GELU','sensitivity_architecture':'9x128 ReLU',
        'novel_subgroup_similarity_less_than':.4}

def now(): return datetime.now(timezone.utc).isoformat()
def atomic_torch(obj,path):
    path=Path(path); tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def module_from(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def source_path(relative):
    choices=[PARENT/'upstream'/relative,PARENT/'upstream/MUBen'/relative,ROOT/'source'/relative]
    for p in choices:
        if p.exists(): return p
    raise FileNotFoundError(choices)

class Model(nn.Module):
    def __init__(self,reg=False,variant='paper'):
        super().__init__();self.reg=reg; layers=[]
        for i in range(8 if variant=='paper' else 9):
            layers.extend([nn.Linear(200 if i==0 else 128,128),nn.GELU() if variant=='paper' else nn.ReLU(),nn.Dropout(.1)])
        self.features=nn.Sequential(*layers);self.head=nn.Linear(128,2 if reg else 1)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight);nn.init.constant_(m.bias,.01)
    def forward(self,x):
        o=self.head(self.features(x));return o if self.reg else o.squeeze(-1)

def data(d,split):
    """Inputs are immutable; derived arrays live in this version's feature cache."""
    path=ROOT/'data'/d/f'{split}.csv'
    cache=ROOT/'features'/f'{d}-{split}.npz';cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:return torch.from_numpy(z['x']),torch.from_numpy(z['y']),z['smiles'].tolist(),z['ids']
    generator=rdNormalizedDescriptors.RDKit2DNormalized();xx=[];yy=[];sm=[];ids=[];nan=0
    for i,row in enumerate(csv.DictReader(path.open())):
        ys,mask=ast.literal_eval(row['labels']),ast.literal_eval(row['masks'])
        assert len(ys)==len(mask)==1 and bool(mask[0]),'Unexpected missing/single-task mask'
        if d!='esol':assert ys[0] in [0,1]
        mol=Chem.MolFromSmiles(row['smiles']);assert mol is not None
        v=generator.process(row['smiles']);assert v and v[0]
        x=np.asarray(v[1:],np.float32);assert x.shape==(200,)
        nan+=int(np.isnan(x).sum());x[np.isnan(x)]=0;assert np.isfinite(x).all()
        xx.append(x);yy.append(ys[0]);sm.append(Chem.MolToSmiles(mol, isomericSmiles=True));ids.append(f'{d}/{split}/{i}')
    x=np.asarray(xx,np.float32);y=np.asarray(yy,np.float32);assert np.isfinite(y).all()
    np.savez_compressed(cache,x=x,y=y,smiles=np.asarray(sm),ids=np.asarray(ids))
    save_json(cache.with_suffix('.json'),{'rows':len(y),'source_sha256':digest(path),'nan_features_to_zero':nan,'rdkit':rdBase.rdkitVersion})
    return torch.from_numpy(x),torch.from_numpy(y),sm,np.asarray(ids)

def noise(o): return F.softplus(o[:,1])
def loss(o,y,reg,focal=False):
    if reg:return F.gaussian_nll_loss(o[:,0],y,noise(o)+1e-6)
    if focal:return sigmoid_focal_loss(o,y,alpha=.25,gamma=2,reduction='mean')
    return F.binary_cross_entropy_with_logits(o,y)
def predict(m,x,drop=False):
    m.eval()
    if drop:
        for layer in m.modules():
            if isinstance(layer,nn.Dropout):layer.train()
    with torch.no_grad():o=torch.cat([m(b) for b in x.split(256)])
    m.eval();return o

def train(d,s,variant='paper',focal=False):
    reg=d=='esol';x,y,_,_=data(d,'train');vx,vy,_,_=data(d,'valid')
    shift=float(y.double().mean()) if reg else 0.;scale=float(y.double().std(unbiased=False)) if reg else 1.
    assert scale>0;y=(y-shift)/scale;vy=(vy-shift)/scale
    kind='focal' if focal else variant;folder=ROOT/'models'/d/kind/f'seed-{s}';folder.mkdir(parents=True,exist_ok=True)
    seed_all(s);m=Model(reg,variant);optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01)
    gen=torch.Generator().manual_seed(s);loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=256,shuffle=True,generator=gen)
    meta=folder/'summary.json'
    if meta.exists():
        result=json.loads(meta.read_text());assert result['source_sha256']==digest(__file__)
        m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));m.eval();return m,result
    best=-float('inf');stale=0;hist=[];start=0;elapsed_before=0
    if (folder/'resume.pt').exists():
        r=torch.load(folder/'resume.pt',weights_only=True);assert r['source_sha256']==digest(__file__)
        m.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer']);torch.set_rng_state(r['rng']);gen.set_state(r['loader_rng'])
        best,stale,hist,start,elapsed_before=r['best'],r['stale'],r['history'],r['epoch'],r['elapsed']
    tick=time.perf_counter()
    for epoch in range(start,400):
        m.train();total=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True);l=loss(m(bx),by,reg,focal);assert torch.isfinite(l)
            l.backward();optimizer.step();total+=float(l.detach())*len(by)
        o=predict(m,vx)
        score=-float(torch.mean((o[:,0]-vy)**2).sqrt()) if reg else float(roc_auc_score(vy.numpy(),o.sigmoid().numpy()))
        if score>=best:
            best=score;stale=0;atomic_torch(m.state_dict(),folder/'best.pt');best_epoch=epoch+1
        else:stale+=1
        hist.append({'epoch':epoch+1,'train_loss':total/len(y),'validation_selection':score})
        # Save a restart point every ten epochs and at stopping/end, never only in RAM.
        if (epoch+1)%10==0 or stale>50 or epoch==399:
            atomic_torch(dict(model=m.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),loader_rng=gen.get_state(),best=best,stale=stale,history=hist,epoch=epoch+1,elapsed=elapsed_before+time.perf_counter()-tick,source_sha256=digest(__file__)),folder/'resume.pt')
        if stale>50:break
    elapsed=elapsed_before+time.perf_counter()-tick
    m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));m.eval()
    original=predict(m,vx);clone=Model(reg,variant);clone.load_state_dict(m.state_dict());torch.testing.assert_close(original,predict(clone,vx),rtol=0,atol=0)
    best_epoch=max(r['epoch'] for r in hist if r['validation_selection']==best)
    result={'dataset':d,'seed':s,'variant':kind,'epochs':len(hist),'best_epoch':best_epoch,'validation_selection':best,'training_seconds':elapsed,'target_mean':shift,'target_std':scale,'parameters':sum(p.numel() for p in m.parameters()),'checkpoint_reload_verified':True,'source_sha256':digest(__file__)}
    save_json(folder/'history.json',hist);save_json(meta,result)
    print(json.dumps(result),flush=True);return m,result

def novelty(train_sm,query_sm):
    gen=rdFingerprintGenerator.GetMorganGenerator(**CONFIG['fingerprints'])
    ref=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_sm]
    return np.asarray([1-max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)),ref)) for s in query_sm])

def gnll(y,mu,var):
    v=np.maximum(var,1e-6);return float(.5*np.mean(np.log(v)+(y-mu)**2/v))
def cnll(y,p):
    return float(F.binary_cross_entropy(torch.as_tensor(p,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64)))
def integrate(mu,base,var,reg):
    return (mu,base+var) if reg else (expit(mu/np.sqrt(1+np.pi*var/8)),None)
def score(y,mu,var,reg):return gnll(y,mu,var) if reg else cnll(y,mu)

def fit_posterior(m,d,s,meta,folder):
    x,y,_,_=data(d,'train');vx,vy,_,_=data(d,'valid');reg=d=='esol'
    if not reg:
        theta,cov,details=fit_laplace(m,x,y,vx,vy,folder)
        return cov.numpy(),details
    h=phi(features(m,x)).numpy();vh=phi(features(m,vx)).numpy();o=predict(m,x);vo=predict(m,vx)
    train_var=noise(o).double().numpy();H=h.T@((1/np.maximum(train_var,1e-6))[:,None]*h)
    # Independent automatic differentiation of a small Gaussian mean-head Hessian.
    z=torch.as_tensor(h[:17],dtype=torch.float64);w=torch.zeros(129,dtype=torch.float64);nv=torch.as_tensor(np.maximum(train_var[:17],1e-6));yy=torch.zeros(17,dtype=torch.float64)
    auto=torch.autograd.functional.hessian(lambda q:F.gaussian_nll_loss(z@q,yy,nv,reduction='sum'),w)
    np.testing.assert_allclose(auto.numpy(),z.numpy().T@((1/nv.numpy())[:,None]*z.numpy()),rtol=1e-8,atol=1e-8)
    ys=(vy.numpy()-meta['target_mean'])/meta['target_std'];mu=vo[:,0].numpy();base=noise(vo).numpy();trials=[]
    for a in ALPHA:
        cov=np.linalg.inv(H+a*np.eye(129));v=np.maximum(np.einsum('ij,jk,ik->i',vh,cov,vh),0)
        trials.append({'alpha':a,'validation_nll':gnll(ys,mu,base+v)})
    best=min(trials,key=lambda t:t['validation_nll']);cov=np.linalg.inv(H+best['alpha']*np.eye(129))
    np.savez_compressed(folder/'laplace_posterior.npz',hessian=H,covariance=cov,alpha=best['alpha'])
    return cov,dict(prior_precision=best['alpha'],grid=trials,checks={'gaussian_mean_hessian_matches_autograd':True,'precision_positive_definite':bool(np.linalg.eigvalsh(H+best['alpha']*np.eye(129)).min()>0)})

def swag_class():return module_from('pinned_swag',source_path('muben/uncertainty/swag.py')).SWAModel

def fit_swag(m,d,s,meta,folder):
    tick=time.perf_counter();seed_all(s+31000);m=copy.deepcopy(m);SWAG=swag_class();sw=SWAG(m,k_models=20)
    x,y,_,_=data(d,'train');y=(y-meta['target_mean'])/meta['target_std']
    optimizer=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.01)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=256,shuffle=True)
    for e in range(20):
        m.train()
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True);l=loss(m(bx),by,d=='esol');assert torch.isfinite(l);l.backward();optimizer.step()
        # Original updater stores first params.data.cpu() directly. Copy to avoid CPU aliasing.
        sw.update_parameters(copy.deepcopy(m))
    atomic_torch(sw.state_dict(),folder/'swag.pt')
    return {'fit_seconds':time.perf_counter()-tick,'snapshots':int(sw.n_averaged),'sampling':'pinned blockwise low-rank + diagonal'}

def fit_all():
    assert not (ROOT/'FROZEN.json').exists(),'Fit is already frozen; preserve this experiment version.'
    checks=scientific_checks();save_json(ROOT/'config.json',CONFIG)
    entries=[];t0=time.perf_counter()
    for d in DATASETS:
        x,y,ts,_=data(d,'train');vx,vy,vs,_=data(d,'valid');distance=novelty(ts,vs)
        for group in GROUPS:
            for s in group:train(d,s)
            s=group[0];m,meta=train(d,s);folder=ROOT/'fits'/d/f'seed-{s}';folder.mkdir(parents=True,exist_ok=True)
            reg=d=='esol';vo=predict(m,vx);mu=vo[:,0].numpy().astype(float) if reg else vo.numpy().astype(float)
            base=noise(vo).numpy().astype(float) if reg else np.zeros(len(vy))
            target=(vy.numpy()-meta['target_mean'])/meta['target_std'] if reg else vy.numpy()
            tick=time.perf_counter();cov,detail=fit_posterior(m,d,s,meta,folder);laplace_seconds=time.perf_counter()-tick
            vh=phi(features(m,vx)).numpy();v=np.maximum(np.einsum('ij,jk,ik->i',vh,cov,vh),0)
            corrections={}
            for method in ['Novelty only','Laplace constant','Novelty Laplace']:
                initial=np.zeros_like(v) if method=='Novelty only' else v
                factor=np.ones_like(distance) if method=='Laplace constant' else distance**2
                trials=[]
                for lam in LAMBDA:
                    a,b=integrate(mu,base,initial+lam*factor,reg);trials.append({'lambda':lam,'validation_nll':score(target,a,b,reg)})
                best=min(trials,key=lambda a:a['validation_nll']);corrections[method]={'selected_lambda':best['lambda'],'grid':trials}
            fit={'dataset':d,'seed':s,'laplace':detail,'laplace_fit_seconds':laplace_seconds,'corrections':corrections,'target_mean':meta['target_mean'],'target_std':meta['target_std']}
            if not reg:
                T=nn.Parameter(torch.ones(()));opt=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
                vv=vo.detach();tick=time.perf_counter()
                for _ in range(20):
                    opt.zero_grad();l=F.binary_cross_entropy_with_logits(vv/T,vy);l.backward();opt.step()
                assert T.item()>0;fit['temperature']=float(T.detach());fit['temperature_fit_seconds']=time.perf_counter()-tick
                train(d,s,focal=True)
            fit['swag']=fit_swag(m,d,s,meta,folder)
            train(d,s,variant='source_architecture')
            save_json(folder/'settings.json',fit);entries.append(fit)
            print('Fitted all methods',d,s,flush=True)
    checks['all_fit_tests_unopened']=True
    save_json(ROOT/'FROZEN.json',{'at_utc':now(),'source_sha256':digest(__file__),'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'config':CONFIG,'fits':entries,'checks':checks,'fit_stage_seconds':time.perf_counter()-t0,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    print('ALL SETTINGS FROZEN; TEST EVALUATION STILL PENDING',flush=True)

def reg_metrics(y,mu,var,ids):
    var=np.maximum(np.asarray(var,float),1e-6);y=np.asarray(y,float);mu=np.asarray(mu,float)
    assert y.shape==mu.shape==var.shape and np.isfinite(var).all()
    err=(mu-y)**2;pit=ndtr((y-mu)/np.sqrt(var));q=np.linspace(0,1,21)[1:-1]
    ce=float(np.mean((np.mean(pit[:,None]<=q[None,:],axis=0)-q)**2))
    # Match the current pinned source's N x N broadcast separately; it is not paired calibration.
    allpair=ndtr((y[None,:]-mu[:,None])/np.sqrt(var[:,None]));upstream_ce=float(np.mean([(np.mean(allpair<=t)-t)**2 for t in q]))
    order=np.lexsort((ids,var));risk=np.cumsum(err[order])/np.arange(1,len(y)+1)
    met={'rmse':float(np.sqrt(err.mean())),'mae':float(np.abs(mu-y).mean()),'nll':gnll(y,mu,var),'nll_full':gnll(y,mu,var)+.5*np.log(2*np.pi),'ce_paired':ce,'ce_upstream':upstream_ce,'coverage90':float(np.mean(np.abs(y-mu)<=1.644853626951*np.sqrt(var))),'width90':float(np.mean(2*1.644853626951*np.sqrt(var))),'aurc_mse':float(risk.mean())}
    return met,risk

def class_metrics(y,p,ids):
    met,risk,_=old_metrics(y,p,ids);met['nll_clipped']=met['nll'];met['nll']=cnll(y,p)
    tp=torch.from_numpy(p);ty=torch.from_numpy(y.astype(int))
    met['ece15_positive']=float(binary_calibration_error(tp,ty));met['mce']=float(binary_calibration_error(tp,ty,norm='max'))
    return met,risk

def evaluate_all():
    frozen=json.loads((ROOT/'FROZEN.json').read_text());assert frozen['source_sha256']==digest(__file__) and frozen['protocol_sha256']==digest(ROOT/'PROTOCOL.md')
    results=ROOT/'results';results.mkdir(exist_ok=True);rows=[];audits=[];t0=time.perf_counter()
    for d in DATASETS:
        x,y,ts,_=data(d,'train');vx,vy,vs,_=data(d,'valid');tx,ty,ss,ids=data(d,'test');reg=d=='esol';distance=novelty(ts,ss)
        def scaff(sm,chir):return {MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=chir) for s in sm}
        audits.append({'dataset':d,'train_n':len(y),'valid_n':len(vy),'test_n':len(ty),'train_mean':float(y.mean()),'valid_mean':float(vy.mean()),'test_mean':float(ty.mean()),'train_test_canonical_overlap':len(set(ts)&set(ss)),'train_valid_canonical_overlap':len(set(ts)&set(vs)),'valid_test_canonical_overlap':len(set(vs)&set(ss)),'nonchiral_scaffold_train_test_overlap':len(scaff(ts,False)&scaff(ss,False)),'chiral_scaffold_train_test_overlap':len(scaff(ts,True)&scaff(ss,True)),'novel_test_n':int((distance>.6).sum())})
        for group in GROUPS:
            s=group[0];m,meta=train(d,s);folder=ROOT/'fits'/d/f'seed-{s}';fit=json.loads((folder/'settings.json').read_text());seed_all(s+50000)
            pred={};times={};tick=time.perf_counter();o=predict(m,tx)
            raw=o[:,0].numpy().astype(float) if reg else o.numpy().astype(float);base=noise(o).numpy().astype(float) if reg else np.zeros(len(ty))
            pred['Deterministic']=(raw,base) if reg else (expit(raw),None);times['Deterministic']=time.perf_counter()-tick
            if not reg:
                pred['Temperature']=(expit(raw/fit['temperature']),None)
                fm,_=train(d,s,focal=True);tick=time.perf_counter();pred['Focal Loss']=(predict(fm,tx).sigmoid().numpy().astype(float),None);times['Focal Loss']=time.perf_counter()-tick
            tick=time.perf_counter();samples=[predict(m,tx,drop=True) for _ in range(30)]
            if reg:pred['MC dropout']=(torch.stack([a[:,0] for a in samples]).mean(0).numpy(),torch.stack([noise(a) for a in samples]).mean(0).numpy())
            else:pred['MC dropout']=(torch.stack([a.sigmoid() for a in samples]).mean(0).numpy().astype(float),None)
            times['MC dropout']=time.perf_counter()-tick
            members=[];ens_train=0.;tick=time.perf_counter()
            for member in group:
                mm,mt=train(d,member);members.append(predict(mm,tx));ens_train+=mt['training_seconds']
            if reg:pred['Ensemble 5']=(torch.stack([a[:,0] for a in members]).mean(0).numpy(),torch.stack([noise(a) for a in members]).mean(0).numpy())
            else:pred['Ensemble 5']=(torch.stack([a.sigmoid() for a in members]).mean(0).numpy().astype(float),None)
            times['Ensemble 5 load_and_predict']=time.perf_counter()-tick
            sw=swag_class()(m,k_models=20);sw.load_state_dict(torch.load(folder/'swag.pt',weights_only=True));seed_all(s+70000);tick=time.perf_counter()
            draws=[predict(sw.sample_parameters().module,tx) for _ in range(30)]
            if reg:pred['SWAG']=(torch.stack([a[:,0] for a in draws]).mean(0).numpy(),torch.stack([noise(a) for a in draws]).mean(0).numpy())
            else:pred['SWAG']=(torch.stack([a.sigmoid() for a in draws]).mean(0).numpy().astype(float),None)
            times['SWAG']=time.perf_counter()-tick
            with np.load(folder/'laplace_posterior.npz') as z:cov=z['covariance']
            tick=time.perf_counter();th=phi(features(m,tx)).numpy();v=np.maximum(np.einsum('ij,jk,ik->i',th,cov,th),0)
            pred['Laplace']=integrate(raw,base,v,reg);times['Laplace']=time.perf_counter()-tick
            for method,setting in fit['corrections'].items():
                initial=np.zeros_like(v) if method=='Novelty only' else v
                factor=np.ones_like(v) if method=='Laplace constant' else distance**2
                pred[method]=integrate(raw,base,initial+setting['selected_lambda']*factor,reg)
                if setting['selected_lambda']==0:
                    expected=pred['Deterministic'] if method=='Novelty only' else pred['Laplace'];np.testing.assert_allclose(pred[method][0],expected[0],atol=1e-12)
                    if reg:np.testing.assert_allclose(pred[method][1],expected[1],atol=1e-12)
            smodel,smeta=train(d,s,variant='source_architecture');so=predict(smodel,tx)
            pred['Source architecture']=(so[:,0].numpy(),noise(so).numpy()) if reg else (so.sigmoid().numpy().astype(float),None)
            for name,(mu,var) in pred.items():
                mu=np.asarray(mu,float);target=ty.numpy().astype(float)
                if reg:mu=mu*meta['target_std']+meta['target_mean'];var=np.asarray(var,float)*meta['target_std']**2;met,risk=reg_metrics(target,mu,var,ids)
                else:met,risk=class_metrics(target,mu,ids)
                mask=distance>.6;met['novel_n']=int(mask.sum())
                if mask.any():met['novel_nll']=score(target[mask],mu[mask],var[mask] if reg else None,reg)
                if (~mask).any():met['familiar_nll']=score(target[~mask],mu[~mask],var[~mask] if reg else None,reg)
                row={'dataset':d,'seed':s,'method':name,**met};rows.append(row)
                targetfile=results/f'{d}-{s}-{name.replace(" ","_")}.npz'
                values=dict(labels=target,mean=mu,ids=ids,novelty_distance=distance,risk=risk)
                if reg:values['variance']=var
                np.savez_compressed(targetfile,**values)
            save_json(results/f'{d}-{s}-timing.json',{'inference_seconds':times,'ensemble_training_seconds':ens_train,'baseline_training_seconds':meta['training_seconds'],'notes':'Raw method timings; ensemble includes checkpoint loads, others resident. Do not claim directly comparable inference speedups.'})
            save_json(results/'per_seed_metrics.json',rows);print('Evaluated',d,s,flush=True)
    save_json(results/'split_audit.json',audits)
    save_json(results/'evaluation_summary.json',{'at_utc':now(),'rows':len(rows),'seconds':time.perf_counter()-t0,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'frozen_sha256':digest(ROOT/'FROZEN.json')})
    print('EVALUATION COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit','evaluate'],required=True);args=p.parse_args()
    if args.stage=='fit':fit_all()
    else:evaluate_all()
