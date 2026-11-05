"""Tier-B GIN (2D graph): our Tier-B pipeline (structure of tierb_chemberta.py) with the authors' GIN encoder (muben.model.gin: atom embedding + torch_geometric GIN, jk=cat) and
featuriser (muben.utils.chem.smiles_to_2d_graph) imported. Protocol: PROTOCOL.md (sha256 in PROTOCOL.sha256). Paper settings: 5 layers, 128 hidden, dropout 0.1, batch 128, lr 1e-4, 200 epochs, tolerance 50.
  python tierb_gin.py graphs | pilot --datasets D | worker --splits S --datasets D | freeze | evaluate"""
import os
_cvd=os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
V8=Path(__file__).resolve().parent;WORK=V8.parent;V4=WORK/'expanded_v4';REPO=WORK/'repos'/'MUBen'
sys.path.insert(0,str(V4));sys.path.insert(0,str(REPO));os.environ.setdefault('TIERB_OUT',str(V8))
import tierb_dnn as t                  # hides GPUs on import; restore afterwards
if _cvd is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_cvd
import csv, json, time, copy, argparse, traceback
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch_geometric.nn as pygnn
from torch_geometric.data import Data, Batch as PygBatch
from muben.model.gin import GIN as AuthorsGIN
from muben.utils.chem import smiles_to_2d_graph
e=t.e
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
EPOCHS=int(os.environ.get('TIERB_EPOCHS',200));PATIENCE=50;BATCH=128;LR=1e-4
CODE=e.digest(__file__);OUT=t.OUT

def graphs_for(d):
    """Graphs in scaffold-file row order (train, valid, test), cached. Single-atom molecules get an empty (2,0) edge index."""
    cache=V8/'graphs'/f'{d}.pt'
    if cache.exists():return torch.load(cache,weights_only=False)
    sm=[r['smiles'] for p in ['train','valid','test'] for r in csv.DictReader((V4/'data'/d/f'{p}.csv').open())];out=[];empty=0
    for s in sm:
        atoms,edges=smiles_to_2d_graph(s);ei=torch.tensor(edges,dtype=torch.long)
        if ei.numel()==0:ei=torch.zeros((2,0),dtype=torch.long);empty+=1
        else:ei=ei.t().contiguous()
        out.append(Data(x=torch.tensor(atoms,dtype=torch.long),edge_index=ei))
    cache.parent.mkdir(exist_ok=True);torch.save(out,cache);e.save_json(cache.with_suffix('.json'),{'rows':len(out),'molecules_without_edges':empty,'featuriser':'muben.utils.chem.smiles_to_2d_graph'});return out

class GCtx(t.Ctx):
    """Labels, masks and scaling from tierb_dnn.Ctx; graphs in the same row order and splits."""
    def __init__(self,split,d):
        super().__init__(split,d);g=graphs_for(d);ii=t.split_index(d,split);self.graphs={p:[g[i] for i in idx] for p,idx in ii.items()}

class GNet(nn.Module):
    """Authors' GIN encoder (embedding + GIN stack, sum pooling) with a Tier-B output head: T x L outputs (mean, raw variance) or T logits; BBP head optional."""
    def __init__(self,T,reg,head='plain'):
        super().__init__();self.T,self.reg,self.L=T,reg,(2 if reg else 1);self.bayes=head=='bbp'
        cfg=SimpleNamespace(n_lbs=1,n_tasks=T*self.L,max_atomic_num=100,n_gin_layers=5,d_gin_hidden=128,dropout=0.1,uncertainty_method='none',
                            task_type='regression' if reg else 'classification',bbp_prior_sigma=.1)
        g=AuthorsGIN(cfg);self.emb,self.gnn=g.emb,g.gnn;H=128
        if self.bayes:self.head=t.bbp.BBPOutputLayer(H,T*self.L,bbp_prior_sigma=.1)
        else:self.head=nn.Linear(H,T*self.L);nn.init.xavier_uniform_(self.head.weight);nn.init.constant_(self.head.bias,.01)
    def encoder_parameters(self):return list(self.emb.parameters())+list(self.gnn.parameters())
    def features(self,b):return pygnn.global_add_pool(self.gnn(self.emb(b.x),b.edge_index),b.batch)
    def forward(self,b):
        o=self.head(self.features(b))
        if self.bayes:o,self.kl=o
        return o.view(-1,self.T,self.L) if self.reg else o.view(-1,self.T)

def batch(graphs,idx):return PygBatch.from_data_list([graphs[int(i)] for i in idx]).to(DEV)

def predict(net,graphs,drop=False):
    net.eval()
    if drop:
        for m in net.modules():
            if isinstance(m,nn.Dropout):m.train()
    with torch.no_grad():out=[net(batch(graphs,range(s,min(s+BATCH,len(graphs))))).float().cpu() for s in range(0,len(graphs),BATCH)]
    net.eval();return torch.cat(out)

def folder(c,kind,seed):return OUT/'models'/'GIN'/c.split/c.d/kind/f'seed-{seed}'

def fit(c,kind,seed):
    """kind: det | focal | bbp | sgld (as tierb_dnn.fit)."""
    f=folder(c,kind,seed)
    if (f/'summary.json').exists():
        net=GNet(c.T,c.reg,'bbp' if kind=='bbp' else 'plain').to(DEV);net.load_state_dict(torch.load(f/'best.pt',map_location=DEV,weights_only=True));net.eval();return net,json.loads((f/'summary.json').read_text())
    f.mkdir(parents=True,exist_ok=True);e.seed_all(seed);net=GNet(c.T,c.reg,'bbp' if kind=='bbp' else 'plain').to(DEV)
    opt=torch.optim.AdamW(net.encoder_parameters() if kind=='sgld' else net.parameters(),lr=LR,weight_decay=.01)
    sampler=t.SGLD(net.head.parameters(),lr=LR,norm_sigma=.1) if kind=='sgld' else None
    gr=c.graphs['train'];_,y,m=c.train;y,m=y.to(DEV),m.to(DEV);gen=torch.Generator().manual_seed(seed);steps=(len(gr)+BATCH-1)//BATCH
    def epoch():
        net.train();total=0.;perm=torch.randperm(len(gr),generator=gen)
        for s in range(0,len(gr),BATCH):
            idx=perm[s:s+BATCH];opt.zero_grad(set_to_none=True)
            if sampler:sampler.zero_grad()
            l=t.masked_loss(net(batch(gr,idx)),y[idx.to(DEV)],m[idx.to(DEV)],c.reg,kind=='focal')
            if kind=='bbp':l=l+net.kl/(steps*len(idx))
            assert torch.isfinite(l);l.backward();opt.step()
            if sampler:sampler.step()
            total+=float(l.detach())*len(idx)
        return total/len(gr)
    tick=time.perf_counter();best=-float('inf');stale=0;hist=[]
    for ep in range(EPOCHS):
        q=time.perf_counter();l=epoch();s=t.valid_score(c,predict(net,c.graphs['valid']))
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
         'parameters':sum(p.numel() for p in net.parameters()),'target_mean':c.shift.tolist(),'target_std':c.scale.tolist(),'device':str(torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu'),'code_sha256':CODE}
    e.save_json(f/'history.json',hist);e.save_json(f/'summary.json',out);return net,out

def posthoc(c,seed):
    """Per-task temperature on validation and SWAG (extra epochs, lr x 0.5, k=20) on the deterministic seed, as tierb_chemberta."""
    net,_=fit(c,'det',seed);f=folder(c,'posthoc',seed)
    if (f/'summary.json').exists():return
    f.mkdir(parents=True,exist_ok=True);out={'split':c.split,'dataset':c.d,'seed':seed,'code_sha256':CODE}
    if not c.reg:
        vo=predict(net,c.graphs['valid']);_,vy,vm=c.valid;T=nn.Parameter(torch.ones(c.T));opt=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(vo,vy,vm),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(seed+60000))
        for _ in range(20):
            for bo,by,bm in dl:opt.zero_grad();l=t.masked_loss(bo/T,by,bm,False);l.backward();opt.step()
        assert (T>0).all();out['temperature']=T.detach().tolist()
    tick=time.perf_counter();e.seed_all(seed+31000);m=copy.deepcopy(net);sw=t.SWAG(m,k_models=20)
    opt=torch.optim.AdamW(m.parameters(),lr=LR/2,weight_decay=.01);gr=c.graphs['train'];_,y,mk=c.train;y,mk=y.to(DEV),mk.to(DEV)
    for _ in range(t.EXTRA):
        m.train();perm=torch.randperm(len(gr))
        for s in range(0,len(gr),BATCH):
            idx=perm[s:s+BATCH];opt.zero_grad(set_to_none=True);l=t.masked_loss(m(batch(gr,idx)),y[idx.to(DEV)],mk[idx.to(DEV)],c.reg);assert torch.isfinite(l);l.backward();opt.step()
        sw.update_parameters(m)
    e.atomic_torch(sw.state_dict(),f/'swag.pt');out['swag']={'seconds':time.perf_counter()-tick,'snapshots':int(sw.n_averaged)};e.save_json(f/'summary.json',out)

def run_unit(c,kind,seed):
    if kind=='posthoc':posthoc(c,seed)
    else:fit(c,kind,seed)
def done(split,d,kind,seed):return (OUT/'models'/'GIN'/split/d/kind/f'seed-{seed}'/'summary.json').exists()

def worker(splits,datasets):
    claims=OUT/'claims';claims.mkdir(parents=True,exist_ok=True)
    for split in splits:
        for d in datasets:
            c=None
            for kind,seed in t.units(split,d):
                if done(split,d,kind,seed):continue
                try:os.close(os.open(claims/f'GIN_{split}_{d}_{kind}_{seed}',os.O_CREAT|os.O_EXCL))
                except FileExistsError:continue
                c=c or GCtx(split,d);tick=time.perf_counter()
                try:run_unit(c,kind,seed);print(json.dumps({'unit':[split,d,kind,seed],'seconds':round(time.perf_counter()-tick,1)}),flush=True)
                except Exception:
                    (OUT/'failures').mkdir(exist_ok=True);(OUT/'failures'/f'GIN_{split}_{d}_{kind}_{seed}.txt').write_text(traceback.format_exc());print('FAILED',split,d,kind,seed,flush=True)

def freeze():
    for split in t.SPLITS:
        for d in t.DATASETS:
            mk=OUT/'frozen'/f'GIN_{split}_{d}.json'
            if mk.exists() or not all(done(split,d,k,s) for k,s in t.units(split,d)):continue
            mk.parent.mkdir(exist_ok=True)
            e.save_json(mk,{'at_utc':e.now(),'backbone':'GIN','split':split,'dataset':d,'units':len(t.units(split,d)),'code_sha256':CODE,'protocol_sha256':e.digest(V8/'PROTOCOL.md'),'test_unopened':True});print('FROZEN GIN',split,d,flush=True)

def predictions_gin(c):
    """(method, seed) -> (mean, variance) on the test split, as tierb_eval.predictions_chemberta."""
    import tierb_eval as ev
    single,members=t.seeds(c.split);gr=c.graphs['test'];P={}
    for s in single:
        net,_=fit(c,'det',s);ph=json.loads((folder(c,'posthoc',s)/'summary.json').read_text());o=predict(net,gr);P[('Deterministic',s)]=ev.convert(c,o)
        if not c.reg:
            P[('Temperature',s)]=ev.convert(c,o/torch.tensor(ph['temperature']));P[('Focal Loss',s)]=ev.convert(c,predict(fit(c,'focal',s)[0],gr))
        e.seed_all(s+50000);P[('MC dropout',s)]=ev.pool(c,[predict(net,gr,drop=True) for _ in range(t.N_DRAWS)])
        sw=t.SWAG(net,k_models=20);sw.load_state_dict(torch.load(folder(c,'posthoc',s)/'swag.pt',weights_only=True));e.seed_all(s+70000)
        P[('SWAG',s)]=ev.pool(c,[predict(sw.sample_parameters().module.to(DEV),gr) for _ in range(t.N_DRAWS)])
        b,_=fit(c,'bbp',s);e.seed_all(s+50000);P[('BBP',s)]=ev.pool(c,[predict(b,gr) for _ in range(t.N_DRAWS)])
        g=GNet(c.T,c.reg).to(DEV);outs=[]
        for ep in range(t.SGLD_INTERVAL,t.SGLD_SAMPLES*t.SGLD_INTERVAL+1,t.SGLD_INTERVAL):
            g.load_state_dict(torch.load(folder(c,'sgld',s)/'snapshots'/f'epoch-{ep:02d}.pt',map_location=DEV,weights_only=True));outs.append(predict(g,gr))
        P[('SGLD',s)]=ev.pool(c,outs)
    key='pooled' if c.split=='scaffold' else single[0]
    P[('Ensembles',key)]=ev.pool(c,[predict(fit(c,'det',s)[0],gr) for s in members]);return P

def evaluate(splits,datasets):
    import tierb_eval as ev
    for split in splits:
        for d in datasets:
            m=OUT/'frozen'/f'GIN_{split}_{d}.json'
            if m.exists():
                j=json.loads(m.read_text());assert j['code_sha256']==CODE and j['protocol_sha256']==e.digest(V8/'PROTOCOL.md'),'code or protocol changed after freeze'
    ev.evaluate_backbone('GIN',splits,datasets,GCtx,predictions_gin)

def pilot(datasets):
    rec={'device':torch.cuda.get_device_name() if DEV.type=='cuda' else 'cpu','units':{}}
    for d in datasets:
        c=GCtx('scaffold',d);tick=time.perf_counter();posthoc(c,0);wall=time.perf_counter()-tick
        s=json.loads((folder(c,'det',0)/'summary.json').read_text());ph=json.loads((folder(c,'posthoc',0)/'summary.json').read_text())
        rec['units'][d]={'train_n':len(c.graphs['train']),'epochs':s['epochs'],'best_epoch':s['best_epoch'],'mean_epoch_seconds':s['mean_epoch_seconds'],'det_seconds':s['seconds'],'swag_seconds':ph['swag']['seconds'],
                         'unit_wall_seconds':wall,'parameters':s['parameters'],'validation_selection':s['validation_selection']};print(json.dumps({d:rec['units'][d]}),flush=True)
    e.save_json(V8/'PILOT_UNITS.json',rec|{'test_read':False,'code_sha256':CODE})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['graphs','pilot','worker','freeze','evaluate']);p.add_argument('--splits',nargs='*',default=t.SPLITS);p.add_argument('--datasets',nargs='*',default=t.DATASETS);a=p.parse_args()
    if a.stage=='graphs':
        for d in a.datasets:g=graphs_for(d);print(d,len(g),'graphs; atoms min/max',min(len(x.x) for x in g),max(len(x.x) for x in g),flush=True)
    elif a.stage=='pilot':pilot(a.datasets)
    elif a.stage=='worker':worker(a.splits,a.datasets)
    elif a.stage=='freeze':freeze()
    else:evaluate(a.splits,a.datasets)
