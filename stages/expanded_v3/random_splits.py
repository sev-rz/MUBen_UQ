"""Source-algorithm random splits; reuse frozen v2 implementation unchanged."""
from pathlib import Path
import sys,csv,json,shutil
ROOT=Path(__file__).resolve().parent;V2=ROOT.parent/'expanded_v2'
sys.path.insert(0,str(V2))
import run_expansion as e
from evaluate_saved import reloadable_swag_class
import numpy as np
from sklearn.model_selection import train_test_split

def configure(seed):
    e.ROOT=ROOT/'random'/f'split-{seed}';e.ROOT.mkdir(parents=True,exist_ok=True)
    e.GROUPS=[list(range(seed*100,seed*100+5))]
    e.CONFIG={**e.CONFIG,'groups':e.GROUPS,'split_seed':seed,'adapter_sha256':e.digest(__file__)}
    e.source_path=lambda rel:V2/'source'/rel
    e.swag_class=reloadable_swag_class
    return e.ROOT

def prepare(seed):
    dest=configure(seed)
    if (dest/'SPLITS.json').exists():return
    shutil.copy2(ROOT/'PROTOCOL.md',dest/'PROTOCOL.md');members={}
    (dest/'features').mkdir(exist_ok=True)
    for d in e.DATASETS:
        rows=[];arrays=[];counts=[]
        for split in ['train','valid','test']:
            rr=list(csv.DictReader((V2/'data'/d/f'{split}.csv').open()));rows+=rr;counts.append(len(rr))
            with np.load(V2/'features'/f'{d}-{split}.npz') as z:arrays.append({k:z[k].copy() for k in ['x','y','smiles','ids']})
        joined={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
        ids=np.arange(len(rows));remaining,valid=train_test_split(ids,test_size=counts[1],random_state=seed)
        train,test=train_test_split(remaining,test_size=counts[2],random_state=seed)
        assert len(set(train)&set(valid))==len(set(train)&set(test))==len(set(valid)&set(test))==0
        assert sorted(np.concatenate([train,valid,test]).tolist())==ids.tolist()
        members[d]={}
        for split,ii in [('train',train),('valid',valid),('test',test)]:
            folder=dest/'data'/d;folder.mkdir(parents=True,exist_ok=True)
            with (folder/f'{split}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows([rows[i] for i in ii])
            np.savez_compressed(dest/'features'/f'{d}-{split}.npz',**{k:a[ii] for k,a in joined.items()})
            members[d][split]=joined['ids'][ii].tolist()
    e.save_json(dest/'SPLITS.json',{'seed':seed,'membership':members,'ordering':'original train then valid then test','adapter_sha256':e.digest(__file__)})

if __name__=='__main__':
    for seed in [0,1,2]:
        prepare(seed)
        if not (e.ROOT/'FROZEN.json').exists():e.fit_all()
    for seed in [0,1,2]:
        configure(seed)
        if not (e.ROOT/'results/evaluation_summary.json').exists():e.evaluate_all()
    e.save_json(ROOT/'RANDOM_COMPLETE.json',{'seeds':[0,1,2],'rows':sum(len(json.loads((ROOT/'random'/f'split-{s}'/'results/per_seed_metrics.json').read_text())) for s in [0,1,2]),'source_sha256':e.digest(__file__)})
