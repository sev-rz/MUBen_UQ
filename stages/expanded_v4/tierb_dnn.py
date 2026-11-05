"""Tier-B DNN-rdkit: masked multi-task training and all MUBen UQ methods on scaffold and random splits.

Extends expanded_v2/run_expansion.py (trunk, descriptors, SWAG, per-task metrics) and expanded_v3
(BBP/SGLD heads, random-split algorithm). Source rules followed (pinned f0ec556, see DECISIONS.md):
masked loss sum/mask.sum(); per-task metrics then macro average; regression labels standardised with
training mean/population SD; validation selection by ROC-AUC (classification), RMSE, or MAE for QM7.
Each fit unit writes summary.json last (completion marker); a unit is never rerun once complete.

  python tierb_dnn.py prep                      features for all datasets
  python tierb_dnn.py worker --splits S [...]   claim and run units until none are left
  python tierb_dnn.py freeze                    FROZEN marker for every complete split/dataset
  python tierb_dnn.py evaluate                  test metrics only where FROZEN exists
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, csv, ast, json, time, copy, shutil, argparse, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent; V2=WORK/'expanded_v2'; V3=WORK/'expanded_v3'
sys.path.insert(0,str(V2))
import run_expansion as e
from evaluate_saved import reloadable_swag_class
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision.ops import sigmoid_focal_loss
from descriptastorus.descriptors import rdNormalizedDescriptors
torch.set_num_threads(1)
SRC=ROOT/'upstream_src/src/muben/uncertainty'
bbp=e.module_from('v4_bbp',SRC/'bbp.py')
SGLD=e.module_from('v4_sgld',SRC/'sgld.py').SGLDOptimizer
SWAG=reloadable_swag_class()

CLS=['bace','bbbp','clintox','tox21']; REG=['esol','freesolv','lipo','qm7']; DATASETS=CLS+REG
SPLITS=['scaffold','random-0','random-1','random-2']
OUT=Path(os.environ.get('TIERB_OUT',ROOT))          # smoke tests redirect here
EPOCHS=int(os.environ.get('TIERB_EPOCHS',400)); PATIENCE=50; BATCH=256; LR=2e-4
EXTRA=int(os.environ.get('TIERB_EXTRA',20))         # SWAG epochs; SGLD runs 30 samples x interval 2
SGLD_SAMPLES,SGLD_INTERVAL,N_DRAWS=30,2,30
CODE=e.digest(__file__)

def seeds(split):
    """Single-model seeds and ensemble members (paper: M=10 scaffold pooled, M=3 per random split)."""
    if split=='scaffold':return [0,100,200],list(range(10))
    k=int(split.split('-')[1]);return [100*k],[100*k+j for j in range(3)]

def table(d):
    """All rows in scaffold-file order (train, valid, test); cached descriptors, labels and masks."""
    cache=ROOT/'features'/f'{d}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    gen=rdNormalizedDescriptors.RDKit2DNormalized();xs,ys,ms,parts,ids=[],[],[],[],[];nan=0;flag_false=[];unparsed=[]
    for k,part in enumerate(['train','valid','test']):
        for i,row in enumerate(csv.DictReader((ROOT/'data'/d/f'{part}.csv').open())):
            y=np.asarray(ast.literal_eval(row['labels']),np.float64);m=np.asarray(ast.literal_eval(row['masks']),bool)
            # Source (utils/chem.py) keeps process()[1:] whatever the success flag, NaN -> 0; record flagged rows.
            # RDKit 2024.09 rejects some molecules older RDKit parsed (e.g. hypervalent Al): all-NaN -> 0 (DECISIONS D4).
            v=gen.process(row['smiles'])
            if v is None:v=[False]+[np.nan]*200;unparsed.append(f'{part}/{i}')
            elif not v[0]:flag_false.append(f'{part}/{i}')
            x=np.asarray(v[1:],np.float32);assert x.shape==(200,)
            nan+=int(np.isnan(x).sum());x[np.isnan(x)]=0;assert np.isfinite(x).all()
            y[~m]=0;xs.append(x);ys.append(y);ms.append(m);parts.append(k);ids.append(f'{d}/{part}/{i}')
    t={'x':np.stack(xs),'y':np.stack(ys).astype(np.float32),'m':np.stack(ms),'part':np.asarray(parts),'ids':np.asarray(ids)}
    assert np.isfinite(t['y']).all() and t['m'].any(0).all()
    cache.parent.mkdir(exist_ok=True);np.savez_compressed(cache,**t)
    e.save_json(cache.with_suffix('.json'),{'rows':len(parts),'tasks':int(t['y'].shape[1]),'nan_features_to_zero':nan,'observed_labels':int(t['m'].sum()),'descriptor_flag_false':flag_false,'rdkit_unparsed_zero_features':unparsed})
    return t

def split_index(d,split):
    """Scaffold = authors' files; random-k = expanded_v3 algorithm (original sizes, random_state k)."""
    part=table(d)['part']
    if split=='scaffold':return {p:np.where(part==k)[0] for k,p in enumerate(['train','valid','test'])}
    k=int(split.split('-')[1]);ids=np.arange(len(part))
    remaining,valid=train_test_split(ids,test_size=int((part==1).sum()),random_state=k)
    train,test=train_test_split(remaining,test_size=int((part==2).sum()),random_state=k)
    return {'train':train,'valid':valid,'test':test}

class Ctx:
    """Tensors for one split/dataset; regression labels standardised on observed training labels."""
    def __init__(self,split,d):
        self.split,self.d,self.reg=split,d,d in REG;t=table(d);ii=split_index(d,split);self.T=t['y'].shape[1]
        tr=t['y'][ii['train']].astype(np.float64);tr[~t['m'][ii['train']]]=np.nan
        self.shift=np.nanmean(tr,0) if self.reg else np.zeros(self.T);self.scale=np.nanstd(tr,0) if self.reg else np.ones(self.T)
        self.scale=np.where(self.scale==0,1.,self.scale)
        for p,i in ii.items():
            y=(t['y'][i]-self.shift)/self.scale;y[~t['m'][i]]=0
            setattr(self,p,(torch.from_numpy(t['x'][i]),torch.from_numpy(y.astype(np.float32)),torch.from_numpy(t['m'][i].astype(np.float32))))
        self.ids={p:t['ids'][i] for p,i in ii.items()};self.raw_test=t['y'][ii['test']].astype(float);self.mask_test=t['m'][ii['test']]
        self.select='mae' if d=='qm7' else ('rmse' if self.reg else 'roc_auc')

class Net(e.Model):
    """v2 trunk (8x128 GELU, dropout 0.1); output T tasks x (mean, raw variance) or T logits; BBP head optional."""
    def __init__(self,T,reg,head='plain'):
        super().__init__(reg);self.T=T;self.L=2 if reg else 1;self.bayes=head=='bbp'
        if self.bayes:self.head=bbp.BBPOutputLayer(128,T*self.L,bbp_prior_sigma=.1)
        elif T>1:
            self.head=nn.Linear(128,T*self.L);nn.init.xavier_uniform_(self.head.weight);nn.init.constant_(self.head.bias,.01)
    def forward(self,x):
        o=self.head(self.features(x))
        if self.bayes:o,self.kl=o
        return o.view(-1,self.T,self.L) if self.reg else o.view(-1,self.T)

def masked_loss(o,y,m,reg,focal=False):
    if reg:l=F.gaussian_nll_loss(o[...,0],y,F.softplus(o[...,1])+1e-6,reduction='none')
    elif focal:l=sigmoid_focal_loss(o,y,alpha=.25,gamma=2,reduction='none')
    else:l=F.binary_cross_entropy_with_logits(o,y,reduction='none')
    return (l*m).sum()/m.sum()

def valid_score(c,o):
    """Source selection metric, macro over tasks, larger is better."""
    _,y,m=c.valid;vals=[]
    for j in range(c.T):
        k=m[:,j].bool();yy=y[k,j].numpy()
        if c.reg:
            err=(o[k,j,0].numpy()-yy)*c.scale[j];vals.append(-(np.sqrt(np.mean(err**2)) if c.select=='rmse' else np.mean(np.abs(err))))
        elif 0<yy.mean()<1:vals.append(roc_auc_score(yy,o[k,j].sigmoid().numpy()))
    return float(np.mean(vals))

def folder(c,kind,seed):return OUT/'models'/'DNN'/c.split/c.d/kind/f'seed-{seed}'

def earlier(c,kind,seed):
    """Completed v2/v3 run with an identical configuration (rule 3), or None."""
    if c.d not in ['bace','bbbp','esol']:return None
    base=V2 if c.split=='scaffold' else V3/'random'/f'split-{c.split[-1]}'
    if kind in ['det','focal']:p=base/'models'/c.d/('paper' if kind=='det' else 'focal')/f'seed-{seed}'
    elif kind in ['bbp','sgld'] and c.split=='scaffold' and c.d=='esol':p=V3/'bayesian_models'/kind.upper()/f'seed-{seed}'
    else:return None
    return p if (p/'summary.json').exists() else None

def loader(c,seed):
    x,y,m=c.train;gen=torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,m),batch_size=BATCH,shuffle=True,generator=gen),gen

def fit(c,kind,seed):
    """kind: det | focal | bbp | sgld. Returns the selected network (SGLD: after its sampling epochs)."""
    f=folder(c,kind,seed);net=Net(c.T,c.reg,'bbp' if kind=='bbp' else 'plain')
    if (f/'summary.json').exists():
        net.load_state_dict(torch.load(f/'best.pt',weights_only=True));net.eval();return net,json.loads((f/'summary.json').read_text())
    f.mkdir(parents=True,exist_ok=True);old=earlier(c,kind,seed)
    if old is not None:return reuse(c,kind,seed,old,net,f)
    e.seed_all(seed);net=Net(c.T,c.reg,'bbp' if kind=='bbp' else 'plain')
    opt=torch.optim.AdamW(net.features.parameters() if kind=='sgld' else net.parameters(),lr=LR,weight_decay=.01)
    sampler=SGLD(net.head.parameters(),lr=LR,norm_sigma=.1) if kind=='sgld' else None
    data,gen=loader(c,seed)
    def epoch():
        net.train();total=0.
        for bx,by,bm in data:
            opt.zero_grad(set_to_none=True)
            if sampler:sampler.zero_grad()
            l=masked_loss(net(bx),by,bm,c.reg,kind=='focal')
            if kind=='bbp':l=l+net.kl/(len(data)*len(by))
            assert torch.isfinite(l);l.backward();opt.step()
            if sampler:sampler.step()
            total+=float(l.detach())*len(by)
        return total/len(c.train[1])
    tick=time.perf_counter();best=-float('inf');stale=0;hist=[]
    for ep in range(EPOCHS):
        l=epoch();s=valid_score(c,e.predict(net,c.valid[0]))
        if s>=best:best=s;stale=0;e.atomic_torch(net.state_dict(),f/'best.pt')
        else:stale+=1
        hist.append({'epoch':ep+1,'loss':l,'validation_selection':s})
        if stale>PATIENCE:break
    net.load_state_dict(torch.load(f/'best.pt',weights_only=True))
    if sampler:
        # Source: continue from the best weights with the end-of-training optimizer state; keep every 2nd epoch.
        snap=f/'snapshots';snap.mkdir(exist_ok=True)
        for ep in range(SGLD_SAMPLES*SGLD_INTERVAL):
            epoch()
            if (ep+1)%SGLD_INTERVAL==0:e.atomic_torch(net.state_dict(),snap/f'epoch-{ep+1:02d}.pt')
    out={'split':c.split,'dataset':c.d,'kind':kind,'seed':seed,'epochs':len(hist),'best_epoch':max(h['epoch'] for h in hist if h['validation_selection']==best),
         'validation_selection':best,'select_metric':c.select,'seconds':time.perf_counter()-tick,'target_mean':c.shift.tolist(),'target_std':c.scale.tolist(),'code_sha256':CODE}
    e.save_json(f/'history.json',hist);e.save_json(f/'summary.json',out);return net,out

def reuse(c,kind,seed,old,net,f):
    """Copy an identical earlier fit and verify it on this version's inputs."""
    prev=json.loads((old/'summary.json').read_text());shutil.copy2(old/'best.pt',f/'best.pt')
    net.load_state_dict(torch.load(f/'best.pt',weights_only=True));net.eval()
    check={}
    if c.reg:check['target_std_matches']=bool(np.isclose(prev['target_std'],c.scale[0],rtol=1e-6) and np.isclose(prev['target_mean'],c.shift[0],rtol=1e-6))
    if kind in ['det','focal']:
        # v2/v3 selected regression on standardised RMSE; v4 reports it on the original scale (same ordering).
        s=valid_score(c,e.predict(net,c.valid[0]))/(c.scale[0] if c.reg else 1.);check['validation_selection_matches']=bool(abs(s-prev['validation_selection'])<1e-6)
    else:check['finite_validation_outputs']=bool(torch.isfinite(e.predict(net,c.valid[0])).all())
    assert all(check.values()),(old,check)
    if kind=='sgld':
        (f/'snapshots').mkdir(exist_ok=True)
        for ep in range(SGLD_INTERVAL,SGLD_SAMPLES*SGLD_INTERVAL+1,SGLD_INTERVAL):shutil.copy2(old/'snapshots'/f'epoch-{ep:02d}.pt',f/'snapshots'/f'epoch-{ep:02d}.pt')
    out={'split':c.split,'dataset':c.d,'kind':kind,'seed':seed,'reused_from':str(old.relative_to(WORK)),'earlier_summary':prev,'reuse_checks':check,'target_mean':c.shift.tolist(),'target_std':c.scale.tolist(),'code_sha256':CODE}
    e.save_json(f/'summary.json',out);return net,out

def posthoc(c,seed):
    """Temperature (one per task, validation set, source settings) and SWAG on the deterministic seed."""
    net,_=fit(c,'det',seed);f=folder(c,'posthoc',seed)
    if (f/'summary.json').exists():return
    f.mkdir(parents=True,exist_ok=True);out={'split':c.split,'dataset':c.d,'seed':seed,'code_sha256':CODE}
    if not c.reg:
        vo=e.predict(net,c.valid[0]).detach();_,vy,vm=c.valid;T=nn.Parameter(torch.ones(c.T));opt=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vo,vy,vm),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed+60000))
        for _ in range(20):
            for bo,by,bm in dl:opt.zero_grad();l=masked_loss(bo/T,by,bm,False);l.backward();opt.step()
        assert (T>0).all();out['temperature']=T.detach().tolist()
    tick=time.perf_counter();e.seed_all(seed+31000);m=copy.deepcopy(net);sw=SWAG(m,k_models=20)
    opt=torch.optim.AdamW(m.parameters(),lr=LR/2,weight_decay=.01);x,y,mk=c.train
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,mk),batch_size=BATCH,shuffle=True)
    for _ in range(EXTRA):
        m.train()
        for bx,by,bm in dl:
            opt.zero_grad(set_to_none=True);l=masked_loss(m(bx),by,bm,c.reg);assert torch.isfinite(l);l.backward();opt.step()
        sw.update_parameters(copy.deepcopy(m))
    e.atomic_torch(sw.state_dict(),f/'swag.pt');out['swag']={'seconds':time.perf_counter()-tick,'snapshots':int(sw.n_averaged)}
    e.save_json(f/'summary.json',out)

def units(split,d):
    single,members=seeds(split);u=[('posthoc',s) for s in single]
    u+=[(k,s) for k in (['focal'] if d in CLS else [])+['bbp','sgld'] for s in single]
    return u+[('det',s) for s in members if s not in single]

def run_unit(c,kind,seed):
    if kind=='posthoc':posthoc(c,seed)
    else:fit(c,kind,seed)

def done(split,d,kind,seed):return (OUT/'models'/'DNN'/split/d/kind/f'seed-{seed}'/'summary.json').exists()

def worker(splits,datasets):
    """Claim units with O_EXCL files so parallel 1-thread processes share one queue."""
    claims=OUT/'claims';claims.mkdir(parents=True,exist_ok=True)
    for split in splits:
        for d in datasets:
            c=None
            for kind,seed in units(split,d):
                if done(split,d,kind,seed):continue
                try:os.close(os.open(claims/f'DNN_{split}_{d}_{kind}_{seed}',os.O_CREAT|os.O_EXCL))
                except FileExistsError:continue
                c=c or Ctx(split,d);tick=time.perf_counter()
                try:run_unit(c,kind,seed);print(json.dumps({'unit':[split,d,kind,seed],'seconds':round(time.perf_counter()-tick,1)}),flush=True)
                except Exception:
                    (OUT/'failures').mkdir(exist_ok=True);(OUT/'failures'/f'DNN_{split}_{d}_{kind}_{seed}.txt').write_text(traceback.format_exc());print('FAILED',split,d,kind,seed,flush=True)

def freeze():
    for split in SPLITS:
        for d in DATASETS:
            marker=OUT/'frozen'/f'DNN_{split}_{d}.json'
            if marker.exists() or not all(done(split,d,k,s) for k,s in units(split,d)):continue
            marker.parent.mkdir(exist_ok=True)
            e.save_json(marker,{'at_utc':e.now(),'backbone':'DNN','split':split,'dataset':d,'units':len(units(split,d)),'code_sha256':CODE,
                                'protocol_sha256':e.digest(ROOT/'PROTOCOL.md'),'test_unopened':True});print('FROZEN',split,d,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prep','worker','freeze','evaluate'])
    p.add_argument('--splits',nargs='*',default=SPLITS);p.add_argument('--datasets',nargs='*',default=DATASETS);a=p.parse_args()
    if a.stage=='prep':
        for d in a.datasets:t=table(d);print(d,t['x'].shape,t['y'].shape,int(t['m'].sum()),flush=True)
    elif a.stage=='worker':worker(a.splits,a.datasets)
    elif a.stage=='freeze':freeze()
    else:
        import tierb_eval;tierb_eval.evaluate_dnn(a.splits,a.datasets)
