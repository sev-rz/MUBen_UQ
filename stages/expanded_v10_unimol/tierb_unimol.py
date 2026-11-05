"""Tier-B Uni-Mol (3D): our Tier-B pipeline with the authors' Uni-Mol model (muben.model.UniMol, pre-trained checkpoint), dataset and collator imported (11 conformers per molecule,
one sampled per training step, predictions averaged over all 11 at inference after the output transform, as muben.train.trainer_unimol). Our head (linear, T x L outputs), loop, scoring and
evaluation. Option B (scaffold only): deterministic fits for seeds 0-9 (seeds 0,1,2 = Deterministic, all ten = Ensembles), post-hoc MC dropout and temperature scaling.
Paper settings: lr 5e-5, batch 128, 100 epochs, tolerance 40, linear schedule with 10 % warm-up; authors' trainer defaults also used: AdamW betas (0.9,0.99) eps 1e-6, gradient-norm clipping at 1.
  python tierb_unimol.py pilot | worker --datasets D [--seeds ..] | freeze | evaluate"""
import os
_cvd=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
V10=Path(__file__).resolve().parent;WORK=V10.parent;V4=WORK/'expanded_v4';REPO=WORK/'repos'/'MUBen'
sys.path.insert(0,str(V4));sys.path.insert(0,str(REPO));os.environ.setdefault('TIERB_OUT',str(V10))
import tierb_dnn as t
if _cvd is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_cvd
import json, time, argparse, traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import HfArgumentParser
from muben.dataset import DatasetUniMol, DictionaryUniMol, CollatorUniMol
from muben.model import UniMol
from muben.args import ArgumentsUniMol, ConfigUniMol
e=t.e
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
EPOCHS=int(os.environ.get('TIERB_EPOCHS',100));PATIENCE=40;BATCH=128;LR=5e-5;WARMUP=0.1;GRAD_NORM=1.0;MB=int(os.environ.get('UNIMOL_MB',32));EVALB=int(os.environ.get('UNIMOL_EVALB',16))
CKPT=WORK/'data'/'unimol_ckpt'/'mol_pre_no_h_220816.pt';FEAT=WORK/'data'/'unimol_archive'/'extracted'/'molecular_property_prediction'
CODE=e.digest(__file__);OUT=t.OUT;NSEED=10;SINGLE=[0,1,2]

def units_u(d):return [('det',s) for s in range(NSEED)]+[('posthoc',s) for s in SINGLE]

def make_config(d):
    A=['--data_folder',str(V4/'data'),'--unimol_feature_folder',str(FEAT),'--dataset_name',d,'--checkpoint_path',str(CKPT),'--uncertainty_method','none','--model_name','Uni-Mol','--seed','0']
    (a,)=HfArgumentParser(ArgumentsUniMol).parse_args_into_dataclasses(A);return ConfigUniMol().from_args(a).get_meta().validate()

class UCtx(t.Ctx):
    """Labels, masks, scaling from tierb_dnn.Ctx (scaffold files, row order train/valid/test); Uni-Mol atoms and conformers from the authors' dataset class (same row order, verified in check D)."""
    def __init__(self,split,d):
        assert split=='scaffold';super().__init__(split,d);self.cfg=make_config(d);self.dic=DictionaryUniMol.load();self.dic.add_symbol('[MASK]',is_special=True)
        self.col=CollatorUniMol(self.cfg,atom_pad_idx=self.dic.pad());self.ds={p:DatasetUniMol().prepare(config=self.cfg,partition=p,dictionary=self.dic) for p in ['train','valid','test']}
        for p,n in [('train',len(self.train[1])),('valid',len(self.valid[1])),('test',len(self.test[1]))]:assert len(self.ds[p])==n,(p,len(self.ds[p]),n)
        self._cache={}
    def eval_batches(self,part):
        """Cached collated inference batches (EVALB molecules x 11 conformers each) on CPU."""
        if part not in self._cache:
            ds=self.ds[part];self._cache[part]=[self.col([ds[i] for i in range(s,min(s+EVALB,len(ds)))]) for s in range(0,len(ds),EVALB)]
        return self._cache[part]

class UNet(nn.Module):
    """Authors' UniMol (encoder, pooler, pre-trained checkpoint) with our linear head: T x L outputs (mean, raw variance) or T logits."""
    def __init__(self,c):
        super().__init__();self.T,self.reg,self.L=c.T,c.reg,(2 if c.reg else 1);self.m=UniMol(c.cfg,c.dic);H=c.cfg.encoder_embed_dim
        self.m.output_layer=nn.Linear(H,self.T*self.L);nn.init.xavier_uniform_(self.m.output_layer.weight);nn.init.constant_(self.m.output_layer.bias,.01)
    def forward(self,b):
        o=self.m(b);return o.view(-1,self.T,self.L) if self.reg else o.view(-1,self.T)

def inv_softplus(x):x=x.clamp_min(1e-12);return torch.where(x>20,x,torch.log(torch.expm1(x)))
def agg(o,reg,temp=None):
    """o: [N,11,T(,2)] raw per-conformer outputs -> pseudo-raw [N,T(,2)] whose sigmoid (classification) or mean/softplus-variance (regression) equal the average over conformers of the transformed outputs."""
    if not reg:
        z=o/temp if temp is not None else o;p=torch.sigmoid(z).mean(1).clamp(1e-7,1-1e-7);return torch.log(p/(1-p))
    mu=o[...,0].mean(1);v=F.softplus(o[...,1]).mean(1);return torch.stack([mu,inv_softplus(v)],-1)

def predict_raw(net,c,part,drop=False):
    """Per-conformer raw outputs [N,11,...] on CPU."""
    net.eval()
    if drop:
        for m in net.modules():
            if isinstance(m,nn.Dropout):m.train()
    outs=[]
    with torch.no_grad():
        for b in c.eval_batches(part):
            b=b.to(DEV);o=net(b).float().cpu();outs.append(o.view(-1,c.cfg.n_conformation,*o.shape[1:]))
    net.eval();return torch.cat(outs)
def predict(net,c,part,drop=False,temp=None):return agg(predict_raw(net,c,part,drop),c.reg,temp)

def folder(c,kind,seed):return OUT/'models'/'Uni-Mol'/c.split/c.d/kind/f'seed-{seed}'

def fit(c,kind,seed):
    assert kind=='det';f=folder(c,kind,seed)
    if (f/'summary.json').exists():
        net=UNet(c).to(DEV);net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True));net.eval();return net,json.loads((f/'summary.json').read_text())
    f.mkdir(parents=True,exist_ok=True);e.seed_all(seed);net=UNet(c).to(DEV)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=LR,betas=(0.9,0.99),eps=1e-6,weight_decay=.01)
    ds=c.ds['train'];_,y,m=c.train;y,m=y.to(DEV),m.to(DEV);gen=torch.Generator().manual_seed(seed);spe=(len(ds)+BATCH-1)//BATCH;total=EPOCHS*spe;warm=int(WARMUP*total)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:s/max(1,warm) if s<warm else max(0.,(total-s)/max(1,total-warm)))
    def epoch():
        net.train();tot=0.;perm=torch.randperm(len(ds),generator=gen)
        for s in range(0,len(ds),BATCH):
            idx=perm[s:s+BATCH];yb=y[idx.to(DEV)];mb=m[idx.to(DEV)];tm=mb.sum();opt.zero_grad(set_to_none=True);ls=0.
            for k in range(0,len(idx),MB):   # gradient accumulation over micro-batches: same gradient as one batch of 128 (masked mean over the whole batch), bounded memory
                sl=slice(k,k+MB)
                if mb[sl].sum()==0:continue
                b=c.col([ds[int(i)] for i in idx[sl]]).to(DEV);l=t.masked_loss(net(b),yb[sl],mb[sl],c.reg)*(mb[sl].sum()/tm);assert torch.isfinite(l);l.backward();ls+=float(l.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(),GRAD_NORM);opt.step();sch.step();tot+=ls*len(idx)
        return tot/len(ds)
    tick=time.perf_counter();best=-float('inf');stale=0;hist=[]
    for ep in range(EPOCHS):
        q=time.perf_counter();l=epoch();q1=time.perf_counter()-q;s=t.valid_score(c,predict(net,c,'valid'))
        if s>=best:best=s;stale=0;e.atomic_torch(net.state_dict(),f/'best.pt')
        else:stale+=1
        hist.append({'epoch':ep+1,'loss':l,'validation_selection':s,'train_seconds':q1,'seconds':time.perf_counter()-q})
        if stale>PATIENCE:break
    net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True));net.eval()
    out={'split':c.split,'dataset':c.d,'kind':kind,'seed':seed,'epochs':len(hist),'best_epoch':int(np.argmax([h['validation_selection'] for h in hist]))+1,'validation_selection':best,
         'seconds':time.perf_counter()-tick,'mean_epoch_seconds':float(np.mean([h['seconds'] for h in hist])),'mean_train_seconds':float(np.mean([h['train_seconds'] for h in hist])),
         'parameters':int(sum(p.numel() for p in net.parameters())),'scale':c.scale.tolist(),'device':str(torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu'),'code_sha256':CODE}
    e.save_json(f/'history.json',hist);e.save_json(f/'summary.json',out);return net,out

def posthoc(c,seed):
    """Per-task temperature on validation (classification only), as tierb_chemberta; no SWAG in option B."""
    net,_=fit(c,'det',seed);f=folder(c,'posthoc',seed)
    if (f/'summary.json').exists():return
    f.mkdir(parents=True,exist_ok=True);out={'split':c.split,'dataset':c.d,'seed':seed,'code_sha256':CODE}
    if not c.reg:
        vo=predict_raw(net,c,'valid');_,vy,vm=c.valid;T=nn.Parameter(torch.ones(c.T));opt=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vo,vy,vm),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed+60000))
        for _ in range(20):
            for bo,by,bm in dl:opt.zero_grad();l=t.masked_loss(agg(bo,False,T),by,bm,False);l.backward();opt.step()
        assert (T>0).all();out['temperature']=T.detach().tolist()
    e.save_json(f/'summary.json',out)

def run_unit(c,kind,seed):
    if kind=='posthoc':posthoc(c,seed)
    else:fit(c,kind,seed)
def done(d,kind,seed):return (OUT/'models'/'Uni-Mol'/'scaffold'/d/kind/f'seed-{seed}'/'summary.json').exists()

def worker(datasets,seeds=None):
    claims=OUT/'claims';claims.mkdir(parents=True,exist_ok=True)
    for d in datasets:
        c=None
        for kind,seed in units_u(d):
            if seeds is not None and seed not in seeds:continue
            if done(d,kind,seed):continue
            try:os.close(os.open(claims/f'Uni-Mol_scaffold_{d}_{kind}_{seed}',os.O_CREAT|os.O_EXCL))
            except FileExistsError:continue
            c=c or UCtx('scaffold',d);tick=time.perf_counter()
            try:run_unit(c,kind,seed);print(json.dumps({'unit':['scaffold',d,kind,seed],'seconds':round(time.perf_counter()-tick,1)}),flush=True)
            except Exception:
                (OUT/'failures').mkdir(exist_ok=True);(OUT/'failures'/f'Uni-Mol_scaffold_{d}_{kind}_{seed}.txt').write_text(traceback.format_exc());print('FAILED',d,kind,seed,flush=True)

def freeze():
    for d in t.DATASETS:
        mk=OUT/'frozen'/f'Uni-Mol_scaffold_{d}.json'
        if mk.exists() or not all(done(d,k,s) for k,s in units_u(d)):continue
        mk.parent.mkdir(exist_ok=True)
        e.save_json(mk,{'at_utc':e.now(),'backbone':'Uni-Mol','split':'scaffold','dataset':d,'units':len(units_u(d)),'code_sha256':CODE,'protocol_sha256':e.digest(V10/'PROTOCOL.md'),'test_unopened':True});print('FROZEN Uni-Mol scaffold',d,flush=True)

def predictions_unimol(c):
    """(method, seed) -> (mean, variance) on the test split."""
    import tierb_eval as ev
    P={}
    for s in SINGLE:
        net,_=fit(c,'det',s);ph=json.loads((folder(c,'posthoc',s)/'summary.json').read_text());raw=predict_raw(net,c,'test');P[('Deterministic',s)]=ev.convert(c,agg(raw,c.reg))
        if not c.reg:P[('Temperature',s)]=ev.convert(c,agg(raw,False,torch.tensor(ph['temperature'])))
        e.seed_all(s+50000);P[('MC dropout',s)]=ev.pool(c,[predict(net,c,'test',drop=True) for _ in range(t.N_DRAWS)])
    P[('Ensembles','pooled')]=ev.pool(c,[predict(fit(c,'det',s)[0],c,'test') for s in range(NSEED)]);return P

def evaluate(datasets):
    import tierb_eval as ev
    for d in datasets:
        m=OUT/'frozen'/f'Uni-Mol_scaffold_{d}.json'
        if m.exists():
            j=json.loads(m.read_text());assert j['code_sha256']==CODE and j['protocol_sha256']==e.digest(V10/'PROTOCOL.md'),'code or protocol changed after freeze'
    ev.evaluate_backbone('Uni-Mol',['scaffold'],datasets,UCtx,predictions_unimol)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','freeze','evaluate','check']);p.add_argument('--datasets',nargs='*',default=t.DATASETS);p.add_argument('--seeds',nargs='*',type=int,default=None);a=p.parse_args()
    if a.stage=='worker':worker(a.datasets,a.seeds)
    elif a.stage=='freeze':freeze()
    elif a.stage=='check':
        for d in a.datasets:c=UCtx('scaffold',d);print(d,{p:len(c.ds[p]) for p in c.ds},'T',c.T,'reg',c.reg,flush=True)
    else:evaluate(a.datasets)
