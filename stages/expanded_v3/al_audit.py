"""Audit of the saved active-learning runs (no training; nothing is rewritten).

Recomputes every saved query batch from the previous round's saved pool predictions,
fingerprints and labeled set only. An exact match shows that the selections used no
pool labels. Also checks the shared initial sets and round-0 models, disjoint batches
that stay inside the pool, and the 401-label budget.
"""
import json
import numpy as np
import active_learning as al

B=al.Bench();n=0
valid_test=set()
for split in ('valid','test'):
    with np.load(al.V2/f'features/bace-{split}.npz',allow_pickle=False) as z:valid_test|=set(z['ids'].tolist())
for seed in al.SEEDS:
    base=al.OUT/f'seed-{seed}';init=json.loads((base/'initial.json').read_text())['ids']
    idx={str(i):k for k,i in enumerate(B.ids)};labeled0=[idx[i] for i in init]
    p0=None
    for st in al.STRATEGIES:
        folder=base/st;labeled=list(labeled0);seen=set(init)
        with np.load(folder/'pool_predictions_round0.npz') as z:
            p=z['p']
            if p0 is None:p0=p
            assert np.array_equal(p,p0),'round-0 model must be shared across strategies'
        for r in range(1,al.ROUNDS+1):
            with np.load(folder/f'pool_predictions_round{r-1}.npz') as z:p=z['p'];mask=z['labeled']
            assert mask.sum()==len(labeled) and set(np.flatnonzero(mask))==set(labeled)
            cand=np.setdiff1d(np.arange(len(p)),labeled);ref=np.maximum(B.S[:,labeled].max(1),B.SV.max(1))
            again=[str(B.ids[i]) for i in al.select(B,st,cand,p,ref,seed,r)]
            saved=json.loads((folder/f'query_round{r}.json').read_text())['ids']
            assert again==saved,(seed,st,r);assert len(set(saved))==25 and not set(saved)&seen and not set(saved)&valid_test
            seen|=set(saved);labeled+=[idx[i] for i in saved];n+=1
        assert json.loads((folder/'final.json').read_text())['labels_used']==401
print(f'AL audit passed: {n} batches recomputed exactly from saved predictions (no labels); initial sets and round-0 models shared; batches disjoint, pool-only; 401 labels per run')
