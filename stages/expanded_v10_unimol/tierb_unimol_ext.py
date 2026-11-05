"""Uni-Mol extension units (PROTOCOL_ADDENDUM_ext.md): SWAG, BBP, SGLD, focal loss on the scaffold split, seeds 0-2, in our Tier-B pipeline around the authors' Uni-Mol encoder. Imports tierb_unimol unchanged.
  python tierb_unimol_ext.py worker --datasets D | freeze | evaluate --datasets D | check"""
import os,sys,json,copy,time,argparse,traceback
from pathlib import Path
import numpy as np,torch
from torch import nn
import tierb_unimol as u
t,e=u.t,u.e
DEV=u.DEV;V10=u.V10;OUT=u.OUT;BATCH=u.BATCH;LR=u.LR;MB=u.MB;SINGLE=u.SINGLE
SGLD_SAMPLES=int(os.environ.get('UNIMOL_SGLD_SAMPLES',t.SGLD_SAMPLES));SGLD_INTERVAL=t.SGLD_INTERVAL;N_DRAWS=t.N_DRAWS
CODE=e.digest(__file__);ADD=V10/'PROTOCOL_ADDENDUM_ext.md'
CLS=t.CLS

def units_ext(d):return [('swag',s) for s in SINGLE]+[(k,s) for k in (['focal'] if d in CLS else [])+['bbp','sgld'] for s in SINGLE]

class UNetB(u.UNet):
    """Uni-Mol encoder with a Bayes-by-backprop output head; forward stores the KL term."""
    def __init__(self,c):
        super().__init__(c);H=c.cfg.encoder_embed_dim;self.m.output_layer=t.bbp.BBPOutputLayer(H,self.T*self.L,bbp_prior_sigma=.1)
    def forward(self,b):
        o,self.kl=self.m(b);return o.view(-1,self.T,self.L) if self.reg else o.view(-1,self.T)

def make(c,kind):return (UNetB if kind=='bbp' else u.UNet)(c).to(DEV)
def head_params(net):return list(net.m.output_layer.parameters())
def enc_params(net):
    ids={id(p) for p in head_params(net)};return [p for p in net.parameters() if id(p) not in ids]

def fit(c,kind,seed):
    assert kind in ('focal','bbp','sgld');f=u.folder(c,kind,seed)
    if (f/'summary.json').exists():
        net=make(c,kind);net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True));net.eval();return net,json.loads((f/'summary.json').read_text())
    f.mkdir(parents=True,exist_ok=True);e.seed_all(seed);net=make(c,kind)
    opt=torch.optim.AdamW(enc_params(net) if kind=='sgld' else [p for p in net.parameters() if p.requires_grad],lr=LR,betas=(0.9,0.99),eps=1e-6,weight_decay=.01)
    sampler=t.SGLD(head_params(net),lr=LR,norm_sigma=.1) if kind=='sgld' else None
    ds=c.ds['train'];_,y,m=c.train;y,m=y.to(DEV),m.to(DEV);gen=torch.Generator().manual_seed(seed);steps=(len(ds)+BATCH-1)//BATCH;total=u.EPOCHS*steps;warm=int(u.WARMUP*total)
    sch=None if kind=='sgld' else torch.optim.lr_scheduler.LambdaLR(opt,lambda s:s/max(1,warm) if s<warm else max(0.,(total-s)/max(1,total-warm)))
    def epoch():
        net.train();tot=0.;perm=torch.randperm(len(ds),generator=gen)
        for s in range(0,len(ds),BATCH):
            idx=perm[s:s+BATCH];yb=y[idx.to(DEV)];mb=m[idx.to(DEV)];tm=mb.sum();opt.zero_grad(set_to_none=True);ls=0.
            if sampler:sampler.zero_grad()
            for k in range(0,len(idx),MB):
                sl=slice(k,k+MB)
                if mb[sl].sum()==0:continue
                b=c.col([ds[int(i)] for i in idx[sl]]).to(DEV);l=t.masked_loss(net(b),yb[sl],mb[sl],c.reg,kind=='focal')*(mb[sl].sum()/tm)
                if kind=='bbp':l=l+net.kl/(steps*len(idx))*(len(idx[sl])/len(idx))
                assert torch.isfinite(l);l.backward();ls+=float(l.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(),u.GRAD_NORM);opt.step()
            if sampler:sampler.step()
            if sch:sch.step()
            tot+=ls*len(idx)
        return tot/len(ds)
    tick=time.perf_counter();best=-float('inf');stale=0;hist=[]
    for ep in range(u.EPOCHS):
        q=time.perf_counter();l=epoch();s=t.valid_score(c,u.predict(net,c,'valid'))
        if s>=best:best=s;stale=0;e.atomic_torch(net.state_dict(),f/'best.pt')
        else:stale+=1
        hist.append({'epoch':ep+1,'loss':l,'validation_selection':s,'seconds':time.perf_counter()-q})
        if stale>u.PATIENCE:break
    net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True))
    if sampler:   # as tierb_dnn: continue from the best weights with the end-of-training optimiser state, keep every second epoch
        snap=f/'snapshots';snap.mkdir(exist_ok=True)
        for ep in range(SGLD_SAMPLES*SGLD_INTERVAL):
            epoch()
            if (ep+1)%SGLD_INTERVAL==0:e.atomic_torch(net.state_dict(),snap/f'epoch-{ep+1:02d}.pt')
        net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True))
    out={'split':c.split,'dataset':c.d,'kind':kind,'seed':seed,'epochs':len(hist),'best_epoch':int(np.argmax([h['validation_selection'] for h in hist]))+1,'validation_selection':best,'seconds':time.perf_counter()-tick,
         'parameters':int(sum(p.numel() for p in net.parameters())),'device':str(torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu'),'code_sha256':CODE}
    e.save_json(f/'history.json',hist);e.save_json(f/'summary.json',out);return net,out

def swag_unit(c,seed):
    """SWAG from the Deterministic net of this seed: EXTRA epochs at lr/2, 20 snapshots (tierb_dnn.posthoc recipe)."""
    f=u.folder(c,'swag',seed)
    if (f/'summary.json').exists():return
    net,_=u.fit(c,'det',seed);f.mkdir(parents=True,exist_ok=True);tick=time.perf_counter();e.seed_all(seed+31000);m=copy.deepcopy(net);sw=t.SWAG(m,k_models=20)
    opt=torch.optim.AdamW(m.parameters(),lr=LR/2,betas=(0.9,0.99),eps=1e-6,weight_decay=.01);ds=c.ds['train'];_,y,mk=c.train;y,mk=y.to(DEV),mk.to(DEV)
    for _ in range(t.EXTRA):
        m.train();perm=torch.randperm(len(ds))
        for s in range(0,len(ds),BATCH):
            idx=perm[s:s+BATCH];yb=y[idx.to(DEV)];mb=mk[idx.to(DEV)];tm=mb.sum();opt.zero_grad(set_to_none=True)
            for k in range(0,len(idx),MB):
                sl=slice(k,k+MB)
                if mb[sl].sum()==0:continue
                b=c.col([ds[int(i)] for i in idx[sl]]).to(DEV);l=t.masked_loss(m(b),yb[sl],mb[sl],c.reg)*(mb[sl].sum()/tm);assert torch.isfinite(l);l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),u.GRAD_NORM);opt.step()
        sw.update_parameters(m)
    e.atomic_torch(sw.state_dict(),f/'swag.pt');e.save_json(f/'summary.json',{'split':c.split,'dataset':c.d,'seed':seed,'swag':{'seconds':time.perf_counter()-tick,'snapshots':int(sw.n_averaged)},'code_sha256':CODE})

def run_unit(c,kind,seed):
    if kind=='swag':swag_unit(c,seed)
    else:fit(c,kind,seed)
def done(d,kind,seed):return (OUT/'models'/'Uni-Mol'/'scaffold'/d/kind/f'seed-{seed}'/'summary.json').exists()

def worker(datasets):
    claims=OUT/'claims_ext';claims.mkdir(parents=True,exist_ok=True)
    for d in datasets:
        c=None
        for kind,seed in units_ext(d):
            if done(d,kind,seed):continue
            try:os.close(os.open(claims/f'ext_{d}_{kind}_{seed}',os.O_CREAT|os.O_EXCL))
            except FileExistsError:continue
            c=c or u.UCtx('scaffold',d);tick=time.perf_counter()
            try:run_unit(c,kind,seed);print(json.dumps({'unit':[d,kind,seed],'seconds':round(time.perf_counter()-tick,1)}),flush=True)
            except Exception:
                (OUT/'failures_ext').mkdir(exist_ok=True);(OUT/'failures_ext'/f'{d}_{kind}_{seed}.txt').write_text(traceback.format_exc());print('FAILED',d,kind,seed,flush=True)

def freeze():
    for d in t.DATASETS:
        mk=OUT/'frozen_ext'/f'Uni-Mol_scaffold_{d}.json'
        if mk.exists() or not all(done(d,k,s) for k,s in units_ext(d)):continue
        mk.parent.mkdir(exist_ok=True);e.save_json(mk,{'at_utc':e.now(),'backbone':'Uni-Mol','split':'scaffold','dataset':d,'units':len(units_ext(d)),'code_sha256':CODE,'base_code_sha256':u.CODE,'addendum_sha256':e.digest(ADD),'test_unopened_for_ext':True});print('FROZEN ext',d,flush=True)

def predictions_all(c):
    """Base methods (unchanged base code) plus SWAG, BBP, SGLD and Focal Loss."""
    import tierb_eval as ev
    P=u.predictions_unimol(c)
    for s in SINGLE:
        if not c.reg:P[('Focal Loss',s)]=ev.convert(c,u.agg(u.predict_raw(fit(c,'focal',s)[0],c,'test'),False))
        net,_=u.fit(c,'det',s);sw=t.SWAG(net,k_models=20);sw.load_state_dict(torch.load(u.folder(c,'swag',s)/'swag.pt',weights_only=True));e.seed_all(s+70000)
        P[('SWAG',s)]=ev.pool(c,[u.predict(sw.sample_parameters().module.to(DEV),c,'test') for _ in range(N_DRAWS)])
        b,_=fit(c,'bbp',s);e.seed_all(s+50000);P[('BBP',s)]=ev.pool(c,[u.predict(b,c,'test') for _ in range(N_DRAWS)])
        g=u.UNet(c).to(DEV);outs=[]
        for ep in range(SGLD_INTERVAL,SGLD_SAMPLES*SGLD_INTERVAL+1,SGLD_INTERVAL):
            g.load_state_dict(torch.load(u.folder(c,'sgld',s)/'snapshots'/f'epoch-{ep:02d}.pt',map_location=DEV,weights_only=True));outs.append(u.predict(g,c,'test'))
        P[('SGLD',s)]=ev.pool(c,outs)
    return P

def evaluate(datasets):
    import tierb_eval as ev
    for d in datasets:
        m=OUT/'frozen_ext'/f'Uni-Mol_scaffold_{d}.json'
        if m.exists():
            j=json.loads(m.read_text());assert j['code_sha256']==CODE and j['addendum_sha256']==e.digest(ADD) and j['base_code_sha256']==u.CODE,'code or addendum changed after freeze'
    ev.evaluate_backbone('Uni-Mol',['scaffold'],datasets,u.UCtx,predictions_all)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['worker','freeze','evaluate','check']);p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args()
    if a.stage=='worker':worker(a.datasets)
    elif a.stage=='freeze':freeze()
    elif a.stage=='evaluate':evaluate(a.datasets)
    else:
        for d in a.datasets:print(d,units_ext(d))
