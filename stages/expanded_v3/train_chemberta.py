"""Full, resumable CPU fine-tuning selected solely by the training-only pilot.

Run fit before evaluate. Training never reads held-out test data. Checkpoints
include AdamW and both PyTorch RNG streams; restart at the last saved epoch.
"""
from chemberta import *
SEEDS=[0,100,200]

def decision():
    p=json.loads((ROOT/'CHEMBERTA_PILOT.json').read_text())
    assert p['test_read'] is False and p['peak_rss_mib']<2500 and p['estimated_max_seconds_three_seeds']<6*3600
    path=ROOT/'CHEMBERTA_DECISION.json'
    if not path.exists():e.save_json(path,{'at_utc':e.now(),'scope':'three full fine-tunes on BACE; all backbone parameters trainable','seeds':SEEDS,'basis':'pilot runtime and memory only','pilot':p,'max_epochs':200,'patience_strictly_greater':40,'test_results_available':False,'code_sha256':e.digest(__file__)})

def fit():
    decision();path=ROOT/'chemberta_checkpoint';tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    tx,ty,ta=load_split('bace','train',tok);vx,vy,va=load_split('bace','valid',tok)
    for seed in SEEDS:
        folder=ROOT/'chemberta_models'/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():continue
        e.seed_all(seed);model=ChemModel(path);opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);gen=torch.Generator().manual_seed(seed)
        hist=[];best=-float('inf');stale=0;before=0.;start=0
        if (folder/'resume.pt').exists():
            r=torch.load(folder/'resume.pt',weights_only=True);assert r['code_sha256']==e.digest(__file__)
            model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);torch.set_rng_state(r['rng']);gen.set_state(r['loader_rng'])
            hist,best,stale,before,start=r['history'],r['best'],r['stale'],r['seconds'],r['epoch']
        tick=time.perf_counter()
        for ep in range(start,200):
            if stale>40:break
            t=time.perf_counter();loss,steps=epoch(model,tx,ty,tok,opt,gen);logits=predict(model,vx,tok);auc=float(roc_auc_score(vy,logits.sigmoid()))
            if auc>=best:best=auc;stale=0;e.atomic_torch(model.state_dict(),folder/'best.pt')
            else:stale+=1
            hist.append({'epoch':ep+1,'loss':loss,'validation_auc':auc,'seconds':time.perf_counter()-t})
            if (ep+1)%5==0 or stale>40 or ep==199:
                e.atomic_torch(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),loader_rng=gen.get_state(),history=hist,best=best,stale=stale,seconds=before+time.perf_counter()-tick,epoch=ep+1,code_sha256=e.digest(__file__)),folder/'resume.pt')
                e.save_json(folder/'progress.json',{'seed':seed,'epoch':ep+1,'best_validation_auc':best,'stale':stale,'elapsed_seconds':before+time.perf_counter()-tick})
                print(json.dumps(hist[-1]|{'seed':seed}),flush=True)
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));vl=predict(model,vx,tok)
        # Verify exact reload without constructing another pretrained model.
        e.atomic_torch(model.state_dict(),folder/'reload.pt');model.load_state_dict(torch.load(folder/'reload.pt',weights_only=True));torch.testing.assert_close(vl,predict(model,vx,tok),rtol=0,atol=0);(folder/'reload.pt').unlink()
        T=nn.Parameter(torch.ones(()));to=torch.optim.AdamW([T],lr=.01,weight_decay=.01)
        for _ in range(20):to.zero_grad();l=F.binary_cross_entropy_with_logits(vl/T,vy);l.backward();to.step()
        assert T.item()>0
        np.savez_compressed(folder/'validation_predictions.npz',logits=vl.numpy(),labels=vy.numpy())
        e.save_json(folder/'history.json',hist)
        e.save_json(folder/'summary.json',{'seed':seed,'epochs':len(hist),'best_epoch':max(h['epoch'] for h in hist if h['validation_auc']==best),'best_validation_auc':best,'temperature':float(T.detach()),'training_seconds':before+time.perf_counter()-tick,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_reload_exact':True,'train_tokens':ta,'valid_tokens':va,'code_sha256':e.digest(__file__)})
    e.save_json(ROOT/'CHEMBERTA_FROZEN.json',{'at_utc':e.now(),'code_sha256':e.digest(__file__),'protocol_sha256':e.digest(ROOT/'PROTOCOL.md'),'test_unopened':True,'seeds':SEEDS})

def evaluate():
    frozen=json.loads((ROOT/'CHEMBERTA_FROZEN.json').read_text());assert frozen['code_sha256']==e.digest(__file__)
    path=ROOT/'chemberta_checkpoint';tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True);tx,ty,ta=load_split('bace','test',tok);ids=np.asarray([f'bace/test/{i}' for i in range(len(ty))]);rows=[]
    out=ROOT/'chemberta_results';out.mkdir(exist_ok=True)
    for seed in SEEDS:
        folder=ROOT/'chemberta_models'/f'seed-{seed}';summary=json.loads((folder/'summary.json').read_text())
        e.seed_all(seed+80000);m=ChemModel(path);m.load_state_dict(torch.load(folder/'best.pt',weights_only=True));z=predict(m,tx,tok)
        predictions={'Deterministic':z.sigmoid().numpy(),'Temperature':(z/summary['temperature']).sigmoid().numpy(),'MC dropout':torch.stack([predict(m,tx,tok,drop=True).sigmoid() for _ in range(30)]).mean(0).numpy()}
        for name,p in predictions.items():
            met,risk=e.class_metrics(ty.numpy().astype(float),p.astype(float),ids);rows.append({'dataset':'bace','backbone':'ChemBERTa','seed':seed,'method':name,**met});np.savez_compressed(out/f'{seed}-{name.replace(" ","_")}.npz',mean=p,labels=ty.numpy(),ids=ids,risk=risk)
    e.save_json(out/'per_seed_metrics.json',rows);e.save_json(out/'audit.json',{'test_tokens':ta,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'source_sha256':e.digest(__file__)})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['fit','evaluate'],required=True);stage=a.parse_args().stage
    fit() if stage=='fit' else evaluate()
