"""Public pinned ChemBERTa checkpoint, CPU pilot and resumable fine-tuning.

Uses the repository's pooler_output, FP32, effective batch128 via microbatch8.
No held-out labels are read by pilot or training functions.
"""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time,csv,ast,resource,argparse,hashlib,math
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent;V2=PARENT/'expanded_v2'
sys.path.insert(0,str(V2))
import run_expansion as e
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModel
from sklearn.metrics import roc_auc_score
from huggingface_hub import HfApi,snapshot_download

def download():
    info=HfApi(token=False).model_info('DeepChem/ChemBERTa-77M-MLM')
    folder=ROOT/'chemberta_checkpoint'
    snapshot_download('DeepChem/ChemBERTa-77M-MLM',revision=info.sha,token=False,local_dir=str(folder),allow_patterns=['config.json','pytorch_model.bin','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'])
    files={str(p.relative_to(folder)):e.digest(p) for p in folder.rglob('*') if p.is_file() and '.cache' not in str(p)}
    e.save_json(ROOT/'CHEMBERTA_SOURCE.json',{'repo':'DeepChem/ChemBERTa-77M-MLM','revision':info.sha,'files':files,'public_download_no_auth':True})
    return folder

def load_split(d,split,tokenizer):
    rows=list(csv.DictReader((V2/'data'/d/f'{split}.csv').open()))
    sm=[r['smiles'] for r in rows];y=torch.tensor([ast.literal_eval(r['labels'])[0] for r in rows],dtype=torch.float32)
    assert all(ast.literal_eval(r['masks'])[0] for r in rows)
    full=tokenizer(sm,add_special_tokens=True,truncation=False)['input_ids']
    tokens=[a[:512] if len(a)>512 else a for a in full]
    # Let the tokenizer place terminal special tokens correctly when truncation is needed.
    tokens=tokenizer(sm,add_special_tokens=True,truncation=True,max_length=512)['input_ids']
    return tokens,y,{'n':len(rows),'max_tokens_untruncated':max(map(len,full)),'truncated':sum(len(a)>512 for a in full)}

class ChemModel(nn.Module):
    def __init__(self,path):
        super().__init__()
        self.backbone=AutoModel.from_pretrained(str(path),local_files_only=True,attn_implementation='eager',trust_remote_code=False)
        self.head=nn.Linear(self.backbone.config.hidden_size,1)
        nn.init.xavier_uniform_(self.head.weight);nn.init.constant_(self.head.bias,.01)
    def forward(self,batch):
        h=self.backbone(**batch).pooler_output
        return self.head(h).squeeze(-1)

def batch_tokens(tokens,idx,tokenizer):
    return tokenizer.pad({'input_ids':[tokens[int(i)] for i in idx]},padding=True,return_tensors='pt')

def predict(model,tokens,tokenizer,drop=False):
    model.eval()
    if drop:
        for m in model.modules():
            if isinstance(m,nn.Dropout):m.train()
    out=[]
    with torch.no_grad():
        for start in range(0,len(tokens),8):out.append(model(batch_tokens(tokens,range(start,min(start+8,len(tokens))),tokenizer)))
    model.eval();return torch.cat(out)

def epoch(model,tokens,y,tokenizer,opt,gen):
    model.train();perm=torch.randperm(len(y),generator=gen);total=0.;steps=0
    for start in range(0,len(y),128):
        ids=perm[start:start+128];opt.zero_grad(set_to_none=True)
        for micro in ids.split(8):
            logits=model(batch_tokens(tokens,micro,tokenizer));loss=F.binary_cross_entropy_with_logits(logits,y[micro],reduction='sum')/len(ids)
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*len(ids)
        opt.step();steps+=1
    return total/len(y),steps

def pilot():
    path=download();tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    tokens,y,ta=load_split('bace','train',tokenizer);vt,vy,va=load_split('bace','valid',tokenizer)
    e.seed_all(90001);m=ChemModel(path);opt=torch.optim.AdamW(m.parameters(),lr=1e-5,weight_decay=.01);gen=torch.Generator().manual_seed(90001)
    times=[]
    for i in range(2):
        t=time.perf_counter();l,steps=epoch(m,tokens,y,tokenizer,opt,gen);z=predict(m,vt,tokenizer);elapsed=time.perf_counter()-t;times.append(elapsed)
        print({'pilot_epoch':i+1,'seconds':elapsed,'loss':l,'validation_auc':float(roc_auc_score(vy,z.sigmoid()))},flush=True)
    result={'parameters':sum(p.numel() for p in m.parameters()),'epoch_seconds':times,'estimated_max_seconds_three_seeds':3*200*max(times),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'train_tokens':ta,'valid_tokens':va,'test_read':False,'microbatch':8,'effective_batch':128,'representation':'source pooler_output','source_sha256':e.digest(__file__)}
    e.save_json(ROOT/'CHEMBERTA_PILOT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot'],required=True);a=p.parse_args();pilot()
