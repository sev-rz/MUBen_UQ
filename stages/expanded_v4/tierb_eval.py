"""Test evaluation for Tier-B, only behind a FROZEN marker. Per-task metrics with v2's verified
functions, then the source's macro average (single-class test tasks skipped, as upstream).
Multi-sample methods average probabilities, or means and variances (paper's aggregation)."""
import json
import numpy as np
import torch
from torch.nn import functional as F
import tierb_dnn as t
e=t.e
CLS_KEYS={'roc-auc':'roc_auc','prc-auc':'prc_auc_trapezoid','ece':'ece15_positive','mce':'mce','nll':'nll','brier':'brier'}
REG_KEYS={'rmse':'rmse','mae':'mae','nll':'nll','ce':'ce_upstream','ce_paired':'ce_paired'}

def convert(c,o):
    o=o.double()
    if c.reg:return (o[...,0].numpy()*c.scale+c.shift,F.softplus(o[...,1]).numpy()*c.scale**2)
    return o.sigmoid().numpy(),None

def pool(c,outs):
    pairs=[convert(c,o) for o in outs]
    return np.mean([a for a,_ in pairs],0),(np.mean([b for _,b in pairs],0) if c.reg else None)

def macro(c,mu,var):
    per=[]
    for j in range(c.T):
        k=c.mask_test[:,j];y=c.raw_test[k,j];ids=c.ids['test'][k]
        if c.reg:met,_=e.reg_metrics(y,mu[k,j],var[k,j],ids);per.append({a:met[b] for a,b in REG_KEYS.items()})
        elif 0<y.mean()<1:met,_=e.class_metrics(y,mu[k,j],ids);per.append({a:met[b] for a,b in CLS_KEYS.items()})
    return {k:float(np.mean([p[k] for p in per])) for k in per[0]}|{'tasks_scored':len(per)}

def predictions_dnn(c):
    """(method, seed) -> (mean, variance) on the test split, original label scale."""
    single,members=t.seeds(c.split);tx=c.test[0];P={}
    for s in single:
        net,_=t.fit(c,'det',s);ph=json.loads((t.folder(c,'posthoc',s)/'summary.json').read_text())
        o=e.predict(net,tx);P[('Deterministic',s)]=convert(c,o)
        if not c.reg:
            P[('Temperature',s)]=convert(c,o/torch.tensor(ph['temperature']))
            P[('Focal Loss',s)]=convert(c,e.predict(t.fit(c,'focal',s)[0],tx))
        e.seed_all(s+50000);P[('MC dropout',s)]=pool(c,[e.predict(net,tx,drop=True) for _ in range(t.N_DRAWS)])
        sw=t.SWAG(net,k_models=20);sw.load_state_dict(torch.load(t.folder(c,'posthoc',s)/'swag.pt',weights_only=True));e.seed_all(s+70000)
        P[('SWAG',s)]=pool(c,[e.predict(sw.sample_parameters().module,tx) for _ in range(t.N_DRAWS)])
        b,_=t.fit(c,'bbp',s);e.seed_all(s+50000);P[('BBP',s)]=pool(c,[e.predict(b,tx) for _ in range(t.N_DRAWS)])
        g=t.Net(c.T,c.reg);outs=[]
        for ep in range(t.SGLD_INTERVAL,t.SGLD_SAMPLES*t.SGLD_INTERVAL+1,t.SGLD_INTERVAL):
            g.load_state_dict(torch.load(t.folder(c,'sgld',s)/'snapshots'/f'epoch-{ep:02d}.pt',weights_only=True));outs.append(e.predict(g,tx))
        P[('SGLD',s)]=pool(c,outs)
    key='pooled' if c.split=='scaffold' else single[0]
    P[('Ensembles',key)]=pool(c,[e.predict(t.fit(c,'det',s)[0],tx) for s in members])
    return P

def evaluate_dnn(splits,datasets):
    for split in splits:
        for d in datasets:
            m=t.OUT/'frozen'/f'DNN_{split}_{d}.json'
            if m.exists():assert json.loads(m.read_text())['code_sha256']==t.CODE,'code changed after freeze'
    evaluate_backbone('DNN',splits,datasets,t.Ctx,predictions_dnn)

def predictions_chemberta(c):
    """As predictions_dnn, for tierb_chemberta models on the GPU."""
    import tierb_chemberta as cb
    single,members=t.seeds(c.split);tx=c.tokens['test'];P={}
    for s in single:
        net,_=cb.fit(c,'det',s);ph=json.loads((cb.folder(c,'posthoc',s)/'summary.json').read_text())
        o=cb.predict(net,tx);P[('Deterministic',s)]=convert(c,o)
        if not c.reg:
            P[('Temperature',s)]=convert(c,o/torch.tensor(ph['temperature']))
            P[('Focal Loss',s)]=convert(c,cb.predict(cb.fit(c,'focal',s)[0],tx))
        e.seed_all(s+50000);P[('MC dropout',s)]=pool(c,[cb.predict(net,tx,drop=True) for _ in range(t.N_DRAWS)])
        sw=t.SWAG(net,k_models=20);sw.load_state_dict(torch.load(cb.folder(c,'posthoc',s)/'swag.pt',weights_only=True));e.seed_all(s+70000)
        P[('SWAG',s)]=pool(c,[cb.predict(sw.sample_parameters().module.to(cb.DEV),tx) for _ in range(t.N_DRAWS)])
        b,_=cb.fit(c,'bbp',s);e.seed_all(s+50000);P[('BBP',s)]=pool(c,[cb.predict(b,tx) for _ in range(t.N_DRAWS)])
        g=cb.CNet(c.T,c.reg).to(cb.DEV);outs=[]
        for ep in range(t.SGLD_INTERVAL,t.SGLD_SAMPLES*t.SGLD_INTERVAL+1,t.SGLD_INTERVAL):
            g.load_state_dict(torch.load(cb.folder(c,'sgld',s)/'snapshots'/f'epoch-{ep:02d}.pt',map_location=cb.DEV,weights_only=True));outs.append(cb.predict(g,tx))
        P[('SGLD',s)]=pool(c,outs)
    key='pooled' if c.split=='scaffold' else single[0]
    P[('Ensembles',key)]=pool(c,[cb.predict(cb.fit(c,'det',s)[0],tx) for s in members])
    return P

def evaluate_backbone(backbone,splits,datasets,make_ctx,predictions):
    for split in splits:
        for d in datasets:
            marker=t.OUT/'frozen'/f'{backbone}_{split}_{d}.json';res=t.OUT/'results'/backbone/split/d
            if not marker.exists() or (res/'per_seed_metrics.json').exists():continue
            c=make_ctx(split,d);res.mkdir(parents=True,exist_ok=True);rows=[]
            for (method,seed),(mu,var) in predictions(c).items():
                assert np.isfinite(mu).all() and (var is None or (np.isfinite(var).all() and (var>0).all()))
                rows.append({'backbone':backbone,'split':split,'dataset':d,'method':method,'seed':seed,**macro(c,mu,var)})
                arrays=dict(mean=mu,labels=c.raw_test,masks=c.mask_test,ids=c.ids['test'])
                if var is not None:arrays['variance']=var
                np.savez_compressed(res/f'{method.replace(" ","_")}-{seed}.npz',**arrays)
            e.save_json(res/'per_seed_metrics.json',rows);print('EVALUATED',backbone,split,d,len(rows),flush=True)

def evaluate_chemberta(splits,datasets):
    import tierb_chemberta as cb
    for split in splits:
        for d in datasets:
            m=t.OUT/'frozen'/f'ChemBERTa_{split}_{d}.json'
            if m.exists():assert json.loads(m.read_text())['code_sha256']==cb.CODE,'code changed after freeze'
    evaluate_backbone('ChemBERTa',splits,datasets,cb.CCtx,predictions_chemberta)
