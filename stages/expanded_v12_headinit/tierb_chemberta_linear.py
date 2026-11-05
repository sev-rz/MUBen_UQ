"""Tier-B ChemBERTa-77M-MLM on one GPU (FP32, deterministic algorithms): same units, seeds, selection,
masked loss and metrics as tierb_dnn; model, tokenisation and pinned checkpoint from expanded_v3/chemberta.py.
Paper settings: lr 1e-5, batch 128, 200 epochs, early-stopping tolerance 40.

  python tierb_chemberta.py pilot --datasets tox21 lipo   time one real unit each -> CHEMBERTA_PILOT_UNITS.json
  python tierb_chemberta.py worker --splits S --datasets D  claim and run units (processes may share one GPU)
  python tierb_chemberta.py freeze | evaluate
"""
import os
_cvd=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1'
import tierb_dnn as t          # v2 import hides GPUs; restore before CUDA initialises
if _cvd is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_cvd
import sys, csv, json, time, copy, argparse, traceback
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
e=t.e
CKPT=t.V3/'chemberta_checkpoint'
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
EPOCHS=int(os.environ.get('TIERB_EPOCHS',200));PATIENCE=40;BATCH=128;LR=1e-5
TOK=AutoTokenizer.from_pretrained(str(CKPT),local_files_only=True)
CODE=e.digest(__file__)

class CCtx(t.Ctx):
    """Labels, masks and scaling from tierb_dnn.Ctx; SMILES tokens in the same row order."""
    def __init__(self,split,d):
        super().__init__(split,d)
        sm=[r['smiles'] for p in ['train','valid','test'] for r in csv.DictReader((t.ROOT/'data'/d/f'{p}.csv').open())]
        full=TOK(sm,add_special_tokens=True,truncation=False)['input_ids'];enc=TOK(sm,add_special_tokens=True,truncation=True,max_length=512)['input_ids']
        self.truncated=sum(len(a)>512 for a in full);ii=t.split_index(d,split)
        self.tokens={p:[enc[i] for i in idx] for p,idx in ii.items()}

class CNet(nn.Module):
    """Source LinearTransformer: pooler_output -> output layer with T x L heads (BBP optional)."""
    def __init__(self,T,reg,head='plain'):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(str(CKPT),local_files_only=True,attn_implementation='eager')
        H=self.backbone.config.hidden_size;self.T,self.reg,self.L=T,reg,(2 if reg else 1);self.bayes=head=='bbp'
        if self.bayes:self.head=t.bbp.BBPOutputLayer(H,T*self.L,bbp_prior_sigma=.1)
        else:self.head=nn.Linear(H,T*self.L);nn.init.xavier_uniform_(self.head.weight);nn.init.constant_(self.head.bias,.01)
    def forward(self,batch):
        o=self.head(self.backbone(**batch).pooler_output)
        if self.bayes:o,self.kl=o
        return o.view(-1,self.T,self.L) if self.reg else o.view(-1,self.T)

def batch(tokens,idx):
    b=TOK.pad({'input_ids':[tokens[int(i)] for i in idx]},padding=True,return_tensors='pt');return {k:v.to(DEV) for k,v in b.items()}

def predict(net,tokens,drop=False):
    net.eval()
    if drop:
        for m in net.modules():
            if isinstance(m,nn.Dropout):m.train()
    with torch.no_grad():out=[net(batch(tokens,range(s,min(s+BATCH,len(tokens))))).float().cpu() for s in range(0,len(tokens),BATCH)]
    net.eval();return torch.cat(out)

def folder(c,kind,seed):return t.OUT/'models'/'ChemBERTa'/c.split/c.d/kind/f'seed-{seed}'

def fit(c,kind,seed):
    """kind: det | focal | bbp | sgld (as tierb_dnn.fit)."""
    f=folder(c,kind,seed)
    if (f/'summary.json').exists():
        net=CNet(c.T,c.reg,'bbp' if kind=='bbp' else 'plain').to(DEV);net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True));net.eval()
        return net,json.loads((f/'summary.json').read_text())
    f.mkdir(parents=True,exist_ok=True);e.seed_all(seed);net=CNet(c.T,c.reg,'bbp' if kind=='bbp' else 'plain').to(DEV)
    opt=torch.optim.AdamW(net.backbone.parameters() if kind=='sgld' else net.parameters(),lr=LR,weight_decay=.01)
    sampler=t.SGLD(net.head.parameters(),lr=LR,norm_sigma=.1) if kind=='sgld' else None
    tok=c.tokens['train'];_,y,m=c.train;y,m=y.to(DEV),m.to(DEV);gen=torch.Generator().manual_seed(seed);steps=(len(tok)+BATCH-1)//BATCH
    total=EPOCHS*steps;warm=int(-(-total*0.1//1));sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:k/max(1,warm) if k<warm else max(0.,(total-k)/max(1,total-warm)))
    def epoch():
        net.train();total=0.;perm=torch.randperm(len(tok),generator=gen)
        for s in range(0,len(tok),BATCH):
            idx=perm[s:s+BATCH];opt.zero_grad(set_to_none=True)
            if sampler:sampler.zero_grad()
            l=t.masked_loss(net(batch(tok,idx)),y[idx.to(DEV)],m[idx.to(DEV)],c.reg,kind=='focal')
            if kind=='bbp':l=l+net.kl/(steps*len(idx))
            assert torch.isfinite(l);l.backward();opt.step()
            if sampler:sampler.step()
            sch.step()
            total+=float(l.detach())*len(idx)
        return total/len(tok)
    tick=time.perf_counter();best=-float('inf');stale=0;hist=[]
    for ep in range(EPOCHS):
        q=time.perf_counter();l=epoch();s=t.valid_score(c,predict(net,c.tokens['valid']))
        if s>=best:best=s;stale=0;e.atomic_torch(net.state_dict(),f/'best.pt')
        else:stale+=1
        hist.append({'epoch':ep+1,'loss':l,'validation_selection':s,'seconds':time.perf_counter()-q})
        if stale>PATIENCE:break
    net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True))
    if sampler:
        snap=f/'snapshots';snap.mkdir(exist_ok=True)
        for ep in range(t.SGLD_SAMPLES*t.SGLD_INTERVAL):
            epoch()
            if (ep+1)%t.SGLD_INTERVAL==0:e.atomic_torch(net.state_dict(),snap/f'epoch-{ep+1:02d}.pt')
    out={'split':c.split,'dataset':c.d,'kind':kind,'seed':seed,'epochs':len(hist),'best_epoch':max(h['epoch'] for h in hist if h['validation_selection']==best),
         'validation_selection':best,'select_metric':c.select,'seconds':time.perf_counter()-tick,'mean_epoch_seconds':float(np.mean([h['seconds'] for h in hist])),
         'target_mean':c.shift.tolist(),'target_std':c.scale.tolist(),'truncated_smiles':c.truncated,'device':str(torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu'),'code_sha256':CODE}
    e.save_json(f/'history.json',hist);e.save_json(f/'summary.json',out);return net,out

def posthoc(c,seed):
    """Per-task temperature on validation (batch 128) and SWAG (20 epochs, lr x 0.5, k=20) on the deterministic seed."""
    net,_=fit(c,'det',seed);f=folder(c,'posthoc',seed)
    if (f/'summary.json').exists():return
    f.mkdir(parents=True,exist_ok=True);out={'split':c.split,'dataset':c.d,'seed':seed,'code_sha256':CODE}
    if not c.reg:
        vo=predict(net,c.tokens['valid']);_,vy,vm=c.valid;T=nn.Parameter(torch.ones(c.T));opt=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vo,vy,vm),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed+60000))
        for _ in range(20):
            for bo,by,bm in dl:opt.zero_grad();l=t.masked_loss(bo/T,by,bm,False);l.backward();opt.step()
        assert (T>0).all();out['temperature']=T.detach().tolist()
    tick=time.perf_counter();e.seed_all(seed+31000);m=copy.deepcopy(net);sw=t.SWAG(m,k_models=20)
    opt=torch.optim.AdamW(m.parameters(),lr=LR/2,weight_decay=.01);tok=c.tokens['train'];_,y,mk=c.train;y,mk=y.to(DEV),mk.to(DEV)
    for _ in range(t.EXTRA):
        m.train();perm=torch.randperm(len(tok))
        for s in range(0,len(tok),BATCH):
            idx=perm[s:s+BATCH];opt.zero_grad(set_to_none=True);l=t.masked_loss(m(batch(tok,idx)),y[idx.to(DEV)],mk[idx.to(DEV)],c.reg);assert torch.isfinite(l);l.backward();opt.step()
        sw.update_parameters(m)
    e.atomic_torch(sw.state_dict(),f/'swag.pt');out['swag']={'seconds':time.perf_counter()-tick,'snapshots':int(sw.n_averaged)}
    e.save_json(f/'summary.json',out)

def run_unit(c,kind,seed):
    if kind=='posthoc':posthoc(c,seed)
    else:fit(c,kind,seed)

def done(split,d,kind,seed):return (t.OUT/'models'/'ChemBERTa'/split/d/kind/f'seed-{seed}'/'summary.json').exists()

def worker(splits,datasets):
    """Several processes may share the GPU; units are claimed with O_EXCL files as in tierb_dnn."""
    claims=t.OUT/'claims';claims.mkdir(parents=True,exist_ok=True)
    for split in splits:
        for d in datasets:
            c=None
            for kind,seed in t.units(split,d):
                if done(split,d,kind,seed):continue
                try:os.close(os.open(claims/f'ChemBERTa_{split}_{d}_{kind}_{seed}',os.O_CREAT|os.O_EXCL))
                except FileExistsError:continue
                c=c or CCtx(split,d);tick=time.perf_counter()
                try:run_unit(c,kind,seed);print(json.dumps({'unit':[split,d,kind,seed],'seconds':round(time.perf_counter()-tick,1)}),flush=True)
                except Exception:
                    (t.OUT/'failures').mkdir(exist_ok=True);(t.OUT/'failures'/f'ChemBERTa_{split}_{d}_{kind}_{seed}.txt').write_text(traceback.format_exc());print('FAILED',split,d,kind,seed,flush=True)

def freeze():
    for split in t.SPLITS:
        for d in t.DATASETS:
            marker=t.OUT/'frozen'/f'ChemBERTa_{split}_{d}.json'
            if marker.exists() or not all(done(split,d,k,s) for k,s in t.units(split,d)):continue
            marker.parent.mkdir(exist_ok=True)
            e.save_json(marker,{'at_utc':e.now(),'backbone':'ChemBERTa','split':split,'dataset':d,'units':len(t.units(split,d)),'code_sha256':CODE,
                                'protocol_sha256':e.digest(t.ROOT/'PROTOCOL.md'),'test_unopened':True});print('FROZEN ChemBERTa',split,d,flush=True)

def pilot(datasets):
    """Time the first real unit (deterministic seed 0 + temperature + SWAG) per dataset; no test data read."""
    rec={'device':torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu','units':{}}
    for d in datasets:
        c=CCtx('scaffold',d);tick=time.perf_counter();posthoc(c,0);wall=time.perf_counter()-tick
        s=json.loads((folder(c,'det',0)/'summary.json').read_text());ph=json.loads((folder(c,'posthoc',0)/'summary.json').read_text())
        rec['units'][d]={'train_n':len(c.tokens['train']),'epochs':s['epochs'],'best_epoch':s['best_epoch'],'mean_epoch_seconds':s['mean_epoch_seconds'],
                         'det_seconds':s['seconds'],'swag_seconds':ph['swag']['seconds'],'unit_wall_seconds':wall,'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if DEV.type=='cuda' else None}
        print(json.dumps({d:rec['units'][d]}),flush=True)
    e.save_json(t.ROOT/'CHEMBERTA_PILOT_UNITS.json',rec|{'test_read':False,'code_sha256':CODE})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','worker','freeze','evaluate'])
    p.add_argument('--splits',nargs='*',default=t.SPLITS);p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args()
    if a.stage=='pilot':pilot(a.datasets)
    elif a.stage=='worker':worker(a.splits,a.datasets)
    elif a.stage=='freeze':freeze()
    else:
        import tierb_eval;tierb_eval.evaluate_chemberta(a.splits,a.datasets)
