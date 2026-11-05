"""Check D (n0095): is our random split the authors' random split? Authors' routine (assist/dataset_build_from_unimol.py): LMDB rows concatenated train,valid,test (Uni-Mol partitions), two successive train_test_split(random_state=seed) with sizes valid then test.
Ours (tierb_dnn.split_index): same algorithm on scaffold-CSV row order. They coincide only if the molecule at each concatenated position is the same. Compared per dataset: sizes, positionwise molecule identity, and test-set overlap per seed."""
import csv, pickle, lmdb, numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
W=Path('/l/users/sevinch.zoda/schub/projects/muben-laplace/work');A=W/'data/unimol_archive/extracted/molecular_property_prediction';D4=W/'expanded_v4/data'
MAP={'bace':'bace','bbbp':'bbbp','clintox':'clintox','tox21':'tox21','esol':'esol','freesolv':'freesolv','lipo':'lipo','qm7':'qm7dft'}
def can(s):
    m=Chem.MolFromSmiles(s);return Chem.MolToSmiles(m) if m is not None else 'RAW:'+s
def lm(p):
    env=lmdb.open(str(p),subdir=False,readonly=True,lock=False,readahead=False,meminit=False,max_readers=256);t=env.begin();out=[]
    for k in t.cursor().iternext(values=False):
        d=pickle.loads(t.get(k));out.append(d['smi'])
    return out
rows=[]
for d,u in MAP.items():
    L={p:lm(A/u/f'{p}.lmdb') for p in ['train','valid','test']}
    C={p:[r['smiles'] for r in csv.DictReader((D4/d/f'{p}.csv').open())] for p in ['train','valid','test']}
    La=sum([L[p] for p in ['train','valid','test']],[]);Ca=sum([C[p] for p in ['train','valid','test']],[])
    sizes_L=[len(L[p]) for p in ['train','valid','test']];sizes_C=[len(C[p]) for p in ['train','valid','test']]
    r={'dataset':d,'n_lmdb':len(La),'n_csv':len(Ca),'sizes_lmdb':sizes_L,'sizes_csv':sizes_C}
    cl=[can(s) for s in La];cc=[can(s) for s in Ca]
    r['same_multiset']=sorted(cl)==sorted(cc);r['posmatch']=float(np.mean([a==b for a,b in zip(cl,cc)])) if len(cl)==len(cc) else None
    r['raw_posmatch']=float(np.mean([a==b for a,b in zip(La,Ca)])) if len(La)==len(Ca) else None
    for k in [0,1,2]:
        # authors' routine on LMDB order
        n=len(La);ids=np.arange(n);rem,v=train_test_split(ids,test_size=sizes_L[1],random_state=k);tr,te=train_test_split(rem,test_size=sizes_L[2],random_state=k)
        authors_test=set(cl[i] for i in te) if True else None
        # ours: sizes from scaffold files, scaffold-CSV order
        if len(Ca)==n:
            rem2,v2=train_test_split(ids,test_size=sizes_C[1],random_state=k);tr2,te2=train_test_split(rem2,test_size=sizes_C[2],random_state=k)
            ours_test=set(cc[i] for i in te2);ours_valid=set(cc[i] for i in v2)
            r[f'test_overlap_{k}']=len(authors_test&ours_test)/max(1,len(authors_test|ours_test)) if len(authors_test)==len(ours_test) else None
            r[f'test_jaccard_{k}']=len(authors_test&ours_test)/max(1,len(authors_test|ours_test))
            r[f'valid_jaccard_{k}']=len(set(cl[i] for i in v)&ours_valid)/max(1,len(set(cl[i] for i in v)|ours_valid))
            r[f'same_index_sets_{k}']=bool(np.array_equal(np.sort(te),np.sort(te2)) and np.array_equal(np.sort(v),np.sort(v2)))
    rows.append(r)
df=pd.DataFrame(rows);df.to_csv('check_D.csv',index=False);pd.set_option('display.width',250);pd.set_option('display.max_columns',40);print(df.drop(columns=[c for c in df if c.startswith('test_overlap')]).round(3).to_string(index=False))
