import sys,os,torch
R='/l/users/sevinch.zoda/schub/projects/muben-laplace/work';sys.path.insert(0,R+'/repos/MUBen')
from transformers import HfArgumentParser
from muben.dataset import DatasetUniMol, DictionaryUniMol, CollatorUniMol
from muben.model import UniMol
from muben.args import ArgumentsUniMol, ConfigUniMol
A=['--data_folder',R+'/expanded_v4/data','--unimol_feature_folder',R+'/data/unimol_archive/extracted/molecular_property_prediction','--dataset_name','bace',
   '--checkpoint_path',R+'/data/unimol_ckpt/mol_pre_no_h_220816.pt','--uncertainty_method','none','--model_name','Uni-Mol','--seed','0']
try:
    (a,)=HfArgumentParser(ArgumentsUniMol).parse_args_into_dataclasses(A)
except SystemExit as ex:
    print('ARGS FAIL');raise
cfg=ConfigUniMol().from_args(a).get_meta().validate()
d=DictionaryUniMol.load();d.add_symbol('[MASK]',is_special=True)
tr=DatasetUniMol().prepare(config=cfg,partition='train',dictionary=d)
print('train n',len(tr))
col=CollatorUniMol(cfg,atom_pad_idx=d.pad());b=col([tr[i] for i in range(4)])
print({k:(tuple(v.shape) if hasattr(v,'shape') else v) for k,v in b.__dict__.items()} if hasattr(b,'__dict__') else b)
m=UniMol(cfg,d);print('params',sum(p.numel() for p in m.parameters())/1e6)
m.eval();print('out',m(b).shape)
va=DatasetUniMol().prepare(config=cfg,partition='valid',dictionary=d);bb=col([va[0]]);print('infer batch atoms',tuple(bb.atoms.shape))
