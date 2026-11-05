"""BACE discovery under a fixed labeling budget (PROTOCOL.md, 'Main extension').

Retrospective simulation. Pool = 1210 official training molecules; validation (151) labels
are a shared initial cost; test (152) is read only to score each final model.
Per seed: 100 label-blind random initial labels, then 6 rounds x 25 acquisitions per
strategy; the paper DNN (v2 run_expansion.train, unchanged) is retrained from scratch each
round with the seed schedule 10000+100*seed+round. Pool labels live only in Oracle and are
revealed for a batch only after its IDs are written to disk.

  python active_learning.py --seed S     one process per seed (1 thread); resumable
  python active_learning.py --summarize  curves, query IDs, summary, paired tests, figure
"""
from pathlib import Path
import os,sys,csv,json,time,argparse,itertools
ROOT=Path(__file__).resolve().parent;V2=ROOT.parent/'expanded_v2'
sys.path.insert(0,str(V2))
import run_expansion as e
import numpy as np
import torch
from scipy.stats import rankdata
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*');torch.set_num_threads(1)

OUT=Path(os.environ.get('AL_OUT',ROOT/'results'/'active_learning'))
SEEDS=[0,1,2,3,4];STRATEGIES=['random','greedy','entropy','diversity','balanced']
N_INIT,BATCH,ROUNDS=100,25,6
CONFIG={'pool':'bace official train (1210)','validation':'bace official valid (151), labels revealed at start, used for checkpoint selection',
 'test':'bace official test (152), read only after round 6 for final-model metrics','seeds':SEEDS,'strategies':STRATEGIES,
 'initial':N_INIT,'batch':BATCH,'rounds':ROUNDS,'budget_labels':N_INIT+BATCH*ROUNDS+151,
 'model':'v2 run_expansion.Model paper 8x128 GELU via unchanged run_expansion.train (AdamW 2e-4 wd.01, batch256, max400, stop when stale>50, ties accepted, validation ROC-AUC)',
 'train_seed':'10000+100*seed+round (identical across strategies)','initial_selection':'numpy default_rng(seed).choice(1210,100,replace=False), label-blind',
 'random_strategy_rng':'numpy default_rng([seed,round,7])','fingerprints':e.CONFIG['fingerprints'],
 'scores':{'activity':'deterministic sigmoid output','entropy':'binary predictive entropy of that probability','distance':'1 - max Tanimoto to reference set'},
 'diversity_reference':'all labeled molecules (labeled pool molecules + 151 validation) plus molecules already in the growing batch',
 'balanced':'sum of average-ranks (higher=better) of activity, entropy and distance among remaining candidates; distance rank recomputed after every pick; no weights',
 'ties':'highest score first, then ascending stable integer ID of bace/train/<i>',
 'scaffold':'Murcko scaffold SMILES, includeChirality=False; empty scaffold = acyclic, counted separately',
 'new_scaffold_reference':'nonempty scaffolds of positives among the 100 initial and 151 validation molecules',
 'audc':'mean over rounds 1..6 of cumulative acquired positives (batch-level; within-batch order carries no information)'}

def key(ids):return np.asarray([int(str(i).rsplit('/',1)[1]) for i in ids])
def fps(smiles):
    g=rdFingerprintGenerator.GetMorganGenerator(**e.CONFIG['fingerprints']);return [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
def scaffold(s):return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),includeChirality=False)
def entropy(p):
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12);return -(p*np.log(p)+(1-p)*np.log1p(-p))
def best_first(score,keys):return np.lexsort((keys,-np.asarray(score,float)))

class Oracle:
    """Sole holder of pool labels; reveals a batch only from its saved query file."""
    def __init__(self):
        with np.load(V2/'features/bace-train.npz',allow_pickle=False) as z:self._y=dict(zip(z['ids'].tolist(),z['y'].astype(float).tolist()))
    def reveal(self,query_file):
        q=json.loads(Path(query_file).read_text());return {i:self._y[i] for i in q['ids']}

class Bench:
    """Features, fingerprints and scaffolds; no pool labels."""
    def __init__(self):
        with np.load(V2/'features/bace-train.npz',allow_pickle=False) as z:self.x,self.ids,self.sm=z['x'],z['ids'],z['smiles'].tolist()
        with np.load(V2/'features/bace-valid.npz',allow_pickle=False) as z:self.vx,self.vy,self.vsm=torch.from_numpy(z['x']),torch.from_numpy(z['y']),z['smiles'].tolist()
        self.key=key(self.ids);assert (self.key==np.arange(len(self.ids))).all() and len(self.ids)==1210 and len(self.vy)==151
        f=fps(self.sm);vf=fps(self.vsm)
        self.S=np.asarray([DataStructs.BulkTanimotoSimilarity(a,f) for a in f]);self.SV=np.asarray([DataStructs.BulkTanimotoSimilarity(a,vf) for a in f])
        self.scaf=[scaffold(s) for s in self.sm];self.vscaf=[scaffold(s) for s in self.vsm]

def fit(B,folder,labeled,labels,seed):
    """Unchanged v2 train(): only its data source and output root are redirected."""
    x=torch.from_numpy(B.x[labeled]);y=torch.tensor([labels[B.ids[i]] for i in labeled],dtype=torch.float32)
    def data(d,split):
        assert d=='bace' and split in ('train','valid'),'acquisition never reads test data'
        return (x,y,None,None) if split=='train' else (B.vx,B.vy,None,None)
    e.ROOT=Path(folder);e.data=data;m,meta=e.train('bace',seed);return m,meta

def select(B,strategy,cand,p,maxsim,seed,r):
    cand=np.sort(np.asarray(cand));H=entropy(p)
    if strategy=='random':return np.random.default_rng([seed,r,7]).choice(cand,BATCH,replace=False).tolist()
    if strategy=='greedy':return cand[best_first(p[cand],B.key[cand])][:BATCH].tolist()
    if strategy=='entropy':return cand[best_first(H[cand],B.key[cand])][:BATCH].tolist()
    ms=maxsim.copy();rem=cand.copy();out=[]
    for _ in range(BATCH):
        dist=1-ms[rem]
        score=dist if strategy=='diversity' else rankdata(p[rem])+rankdata(H[rem])+rankdata(dist)
        j=int(rem[best_first(score,B.key[rem])[0]]);out.append(j);rem=rem[rem!=j];ms=np.maximum(ms,B.S[j])
    return out

def run_seed(seed):
    B=Bench();oracle=Oracle();base=OUT/f'seed-{seed}';base.mkdir(parents=True,exist_ok=True);e.save_json(base/'config.json',CONFIG)
    init_file=base/'initial.json'
    initial=np.sort(np.random.default_rng(seed).choice(len(B.ids),N_INIT,replace=False)).tolist()
    if init_file.exists():assert json.loads(init_file.read_text())['ids']==[str(B.ids[i]) for i in initial]
    else:e.save_json(init_file,{'seed':seed,'ids':[str(B.ids[i]) for i in initial],'label_blind':True})
    init_labels=oracle.reveal(init_file)
    for strategy in STRATEGIES:
        folder=base/strategy;folder.mkdir(exist_ok=True)
        if (folder/'final.json').exists():continue
        labeled=list(initial);labels=dict(init_labels);rounds=[];t0=time.perf_counter()
        for r in range(ROUNDS+1):
            mf=base/'round-0' if r==0 else folder/f'round-{r}'
            tick=time.perf_counter();m,meta=fit(B,mf,labeled,labels,10000+100*seed+r);train_s=time.perf_counter()-tick
            p=e.predict(m,torch.from_numpy(B.x)).sigmoid().numpy().astype(float)
            vp=e.predict(m,B.vx).sigmoid().numpy().astype(float)
            np.savez_compressed(folder/f'pool_predictions_round{r}.npz',p=p,ids=B.ids,labeled=np.isin(np.arange(len(p)),labeled))
            rec={'round':r,'n_labeled_pool':len(labeled),'train_seed':10000+100*seed+r,'validation_auc':meta['validation_selection'],'validation_nll':e.cnll(B.vy.numpy().astype(float),vp),'epochs':meta['epochs'],'best_epoch':meta['best_epoch'],'train_seconds':train_s}
            if r==ROUNDS:rounds.append(rec);break
            tick=time.perf_counter();cand=np.setdiff1d(np.arange(len(p)),labeled)
            ref=np.maximum(B.S[:,labeled].max(1),B.SV.max(1))
            chosen=select(B,strategy,cand,p,ref,seed,r+1);qf=folder/f'query_round{r+1}.json'
            ids=[str(B.ids[i]) for i in chosen];assert len(set(chosen))==BATCH and not set(chosen)&set(labeled)
            if qf.exists():assert json.loads(qf.read_text())['ids']==ids,'resumed selection differs from saved query'
            else:e.save_json(qf,{'seed':seed,'strategy':strategy,'round':r+1,'ids':ids,'selection_order':list(range(1,BATCH+1)),'predicted_activity':p[chosen].tolist(),'entropy':entropy(p[chosen]).tolist(),'distance_before_batch':(1-ref[chosen]).tolist(),'labels_seen_by_strategy':len(labels)})
            rec['select_seconds']=time.perf_counter()-tick;rounds.append(rec)
            labels.update(oracle.reveal(qf));labeled+=chosen
        with np.load(V2/'features/bace-test.npz',allow_pickle=False) as z:tx,ty,tid=torch.from_numpy(z['x']),z['y'].astype(float),z['ids']
        tp=e.predict(m,tx).sigmoid().numpy().astype(float);met,_=e.class_metrics(ty,tp,tid)
        np.savez_compressed(folder/'final_test_predictions.npz',p=tp,labels=ty,ids=tid)
        e.save_json(folder/'rounds.json',rounds)
        e.save_json(folder/'final.json',{'seed':seed,'strategy':strategy,'test':met,'wall_seconds':time.perf_counter()-t0,'labels_used':len(labels)+151,'code_sha256':e.digest(__file__)})
        print(json.dumps({'seed':seed,'strategy':strategy,'test_auc':round(met['roc_auc'],4),'seconds':round(time.perf_counter()-t0,1)}),flush=True)

def summarize():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    B=Bench();oracle=Oracle();curves=[];queries=[];finals=[];e.save_json(OUT/'config.json',CONFIG)
    for seed in SEEDS:
        base=OUT/f'seed-{seed}';init=json.loads((base/'initial.json').read_text())['ids'];il=oracle.reveal(base/'initial.json')
        idx={str(i):n for n,i in enumerate(B.ids)}
        known={B.scaf[idx[i]] for i in init if il[i]==1}|{s for s,y in zip(B.vscaf,B.vy.tolist()) if y==1};known.discard('')
        for strategy in STRATEGIES:
            folder=base/strategy;rounds=json.loads((folder/'rounds.json').read_text());fin=json.loads((folder/'final.json').read_text())
            pos=0;scafs=set();acyc=0;row0={'seed':seed,'strategy':strategy}
            curves.append({**row0,'round':0,'acquired':0,'cumulative_positives':0,'distinct_active_scaffolds':0,'new_active_scaffolds':0,'acyclic_actives':0,'validation_auc':rounds[0]['validation_auc']})
            for r in range(1,ROUNDS+1):
                qf=folder/f'query_round{r}.json';lab=oracle.reveal(qf)
                for rank,i in enumerate(json.loads(qf.read_text())['ids'],1):
                    s=B.scaf[idx[i]];y=lab[i];queries.append({**row0,'round':r,'rank_in_batch':rank,'id':i,'label':int(y),'scaffold':s,'smiles':B.sm[idx[i]]})
                    if y==1:
                        pos+=1
                        if s=='':acyc+=1
                        else:scafs.add(s)
                curves.append({**row0,'round':r,'acquired':r*BATCH,'cumulative_positives':pos,'distinct_active_scaffolds':len(scafs),'new_active_scaffolds':len(scafs-known),'acyclic_actives':acyc,'validation_auc':rounds[r]['validation_auc']})
            c=[x for x in curves if x['seed']==seed and x['strategy']==strategy and x['round']>0]
            remaining=sum(1 for i,y in oracle._y.items() if y==1 and i not in il)
            finals.append({**row0,'positives_150':pos,'distinct_active_scaffolds':len(scafs),'new_active_scaffolds':len(scafs-known),'acyclic_actives':acyc,'audc':float(np.mean([x['cumulative_positives'] for x in c])),
                'initial_positives':int(sum(il.values())),'pool_positives_unlabeled_at_start':remaining,'random_expectation_150':150*remaining/(1210-N_INIT),
                'test_auc':fin['test']['roc_auc'],'test_nll':fin['test']['nll'],'test_brier':fin['test']['brier'],'final_validation_auc':rounds[-1]['validation_auc'],'wall_seconds':fin['wall_seconds'],'train_seconds':sum(x['train_seconds'] for x in rounds)})
    END=['positives_150','distinct_active_scaffolds','new_active_scaffolds','acyclic_actives','audc','test_auc','test_nll','wall_seconds']
    summary=[]
    for st in STRATEGIES:
        f=[x for x in finals if x['strategy']==st];row={'strategy':st,'n_seeds':len(f)}
        for k in END:v=np.asarray([x[k] for x in f],float);row[k+'_mean']=float(v.mean());row[k+'_sd']=float(v.std(ddof=1));row[k+'_min']=float(v.min());row[k+'_max']=float(v.max())
        summary.append(row)
    paired=[]
    for ref in ('random','greedy'):
        for st in STRATEGIES:
            if st==ref:continue
            for k in END[:6]:
                d=np.asarray([next(x[k] for x in finals if x['seed']==s and x['strategy']==st)-next(x[k] for x in finals if x['seed']==s and x['strategy']==ref) for s in SEEDS],float)
                flips=[abs(np.mean(d*np.asarray(sg))) for sg in itertools.product([1,-1],repeat=len(d))]
                paired.append({'strategy':st,'reference':ref,'endpoint':k,'mean_difference':float(d.mean()),'sd_difference':float(d.std(ddof=1)),'per_seed':' '.join(f'{v:g}' for v in d),
                    'n_higher':int((d>0).sum()),'n_lower':int((d<0).sum()),'n_tied':int((d==0).sum()),'exact_signflip_p_two_sided':float(np.mean([f>=abs(d.mean())-1e-12 for f in flips]))})
    def w(name,rows):
        with (OUT/name).open('w',newline='') as fh:wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    w('curves_per_seed.csv',curves);w('query_ids.csv',queries);w('per_seed_endpoints.csv',finals);w('summary.csv',summary);w('paired_comparisons.csv',paired)
    colors={'random':'#7f7f7f','greedy':'#d62728','entropy':'#1f77b4','diversity':'#2ca02c','balanced':'#9467bd'}
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,k,lab in [(axes[0],'cumulative_positives','Cumulative acquired actives'),(axes[1],'new_active_scaffolds','New active scaffolds')]:
        for st in STRATEGIES:
            M=np.asarray([[x[k] for x in curves if x['seed']==s and x['strategy']==st] for s in SEEDS],float);n=np.arange(ROUNDS+1)*BATCH
            for row in M:ax.plot(n,row,color=colors[st],alpha=.18,lw=.8)
            ax.plot(n,M.mean(0),color=colors[st],lw=2,label=st,marker='o',ms=3)
        if k=='cumulative_positives':ax.plot(n,n*np.mean([x['random_expectation_150'] for x in finals])/150,'k:',lw=1,label='random expectation')
        ax.set_xlabel('Acquired labels (after 100 initial + 151 validation)');ax.set_ylabel(lab);ax.set_xticks(n);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('BACE fixed-budget retrospective discovery, 5 seeds (thin: per seed; thick: mean)',fontsize=10)
    fig.savefig(OUT/'discovery_curves.png',dpi=160);plt.close(fig)
    for r in summary:print(f"{r['strategy']:9s} pos150 {r['positives_150_mean']:.1f}±{r['positives_150_sd']:.1f} newScaf {r['new_active_scaffolds_mean']:.1f}±{r['new_active_scaffolds_sd']:.1f} audc {r['audc_mean']:.1f} testAUC {r['test_auc_mean']:.3f}±{r['test_auc_sd']:.3f} NLL {r['test_nll_mean']:.3f}")
    for p in paired:
        if p['endpoint'] in ('positives_150','new_active_scaffolds'):print(f"{p['strategy']:9s}-{p['reference']:6s} {p['endpoint']:21s} {p['mean_difference']:+.1f} [{p['per_seed']}] p={p['exact_signflip_p_two_sided']:.3f}")

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int);a.add_argument('--summarize',action='store_true');a=a.parse_args()
    summarize() if a.summarize else run_seed(a.seed)
