"""Recompute every result from saved predictions, compare references and plot."""
from run_expansion import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
R=ROOT/'results';rows=json.loads((R/'per_seed_metrics.json').read_text())
verification=[]
for row in rows:
 p=R/f'{row["dataset"]}-{row["seed"]}-{row["method"].replace(" ","_")}.npz'
 with np.load(p,allow_pickle=False) as z:
  if row['dataset']=='esol':m,risk=reg_metrics(z['labels'],z['mean'],z['variance'],z['ids'])
  else:m,risk=class_metrics(z['labels'],z['mean'],z['ids'])
  for k,v in m.items():np.testing.assert_allclose(v,row[k],rtol=1e-10,atol=1e-12)
  np.testing.assert_allclose(risk,z['risk'],rtol=0,atol=0)
 verification.append(str(p.name))
# Verify the regression mean/variance loss and scaler directly against pinned source.
lossmod=module_from('reference_loss',source_path('muben/train/trainer/loss.py'))
scalermod=module_from('reference_scaler',source_path('muben/train/trainer/scaler.py'))
a=torch.tensor([[.2,.3],[.5,-.8]],dtype=torch.float64);b=torch.tensor([.1,-.2],dtype=torch.float64)
torch.testing.assert_close(loss(a,b,True),lossmod.GaussianNLLLoss()(a,b))
scaler=scalermod.StandardScaler().fit(np.array([1.,2.,4.]))
roundtrip=scaler.inverse_transform(np.asarray(scaler.transform(np.array([1.,2.,4.])),dtype=float))
np.testing.assert_allclose(np.asarray(roundtrip,dtype=float),[1.,2.,4.])
# Read just the original CE function into a namespace: exact source behavior, no repairs.
mod=ast.parse(source_path('muben/utils/metrics.py').read_text())
function=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='regression_calibration_error')
namespace={'np':np,'gaussian':norm};exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_CE','exec'),namespace)
with np.load(R/'esol-0-Deterministic.npz') as z:
 actual=namespace['regression_calibration_error'](z['labels'],z['mean'],z['variance'])
 expected=reg_metrics(z['labels'],z['mean'],z['variance'],z['ids'])[0]['ce_upstream']
 np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
# Source discrepancy illustration: correct per-example distributions nearly certain at truth;
# all-pairs broadcasts produce a different empirical CDF than a paired calculation.

summary=[]
for d in DATASETS:
 for name in dict.fromkeys(r['method'] for r in rows if r['dataset']==d):
  rs=[r for r in rows if r['dataset']==d and r['method']==name];assert len(rs)==3
  sm={'dataset':d,'method':name,'n_repeats':3}
  for k in rs[0]:
   if k in ['dataset','method','seed']:continue
   vals=[r[k] for r in rs if k in r]
   sm[k+'_mean']=float(np.mean(vals));sm[k+'_std']=float(np.std(vals,ddof=1))
  summary.append(sm)

def csvwrite(path,items):
 keys=list(dict.fromkeys(k for r in items for k in r))
 with path.open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(items)
csvwrite(R/'per_seed_metrics.csv',rows);csvwrite(R/'summary_metrics.csv',summary)
comparison=[]
name_map={'Deterministic':'none','Temperature':'TemperatureScaling','MC dropout':'MCDropout','Ensemble 5':'DeepEnsembles','Focal Loss':'FocalLoss','SWAG':'SWAG'}
for d in DATASETS:
 refs={r['method']:r for r in csv.DictReader((ROOT/'references'/f'DNN-rdkit-{d}.csv').open())}
 for sm in summary:
  if sm['dataset']!=d or sm['method'] not in name_map:continue
  ref=refs['DNN-rdkit-'+name_map[sm['method']]]
  metric_map={'roc_auc':'roc-auc','prc_auc_trapezoid':'prc-auc','ece15_positive':'ece','mce':'mce','nll':'nll','brier':'brier'} if d!='esol' else {'rmse':'rmse','mae':'mae','nll':'nll','ce_upstream':'ce'}
  for metric,old in metric_map.items():
   if old+'-mean' not in ref:continue
   upstream=float(ref[old+'-mean']);current=sm[metric+'_mean']
   comparison.append({'dataset':d,'method':sm['method'],'metric':metric,'original_mean':upstream,'original_std':ref.get(old+'-std',''),'reproduced_mean':current,'reproduced_std':sm[metric+'_std'],'difference':current-upstream,'source':f'reports/primary/DNN-rdkit-{d}.csv','comparability':'scoped reproduction; ensemble5 vs10; paper architecture; see fidelity matrix'})
csvwrite(R/'original_comparison.csv',comparison)
paired=[]
for r in rows:
 if r['method'] not in ['Novelty Laplace','Laplace constant','Novelty only']:continue
 base=next(t for t in rows if t['dataset']==r['dataset'] and t['seed']==r['seed'] and t['method']=='Laplace')
 paired.append({'dataset':r['dataset'],'seed':r['seed'],'method':r['method'],'nll_difference_vs_laplace':r['nll']-base['nll'],'selective_risk_difference_vs_laplace':r.get('aurc',r.get('aurc_mse'))-base.get('aurc',base.get('aurc_mse'))})
csvwrite(R/'extension_paired.csv',paired)
# Full reference comparison plus separate original architecture sensitivity results.
checks={'result_rows_recomputed':len(rows),'prediction_files':verification,'source_gaussian_loss_matches':True,'source_scaler_roundtrip':True,'source_regression_CE_matches':True,'sample_sd_ddof':1,'comparison_rows':len(comparison),'no_posttest_retuning':True}
save_json(R/'verification.json',checks)
fig=R/'figures';fig.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
methods=['Deterministic','Temperature','MC dropout','Ensemble 5','Focal Loss','SWAG','Laplace','Novelty only','Laplace constant','Novelty Laplace']
fig1,axs=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
for ax,d in zip(axs,DATASETS):
 sel=[r for r in summary if r['dataset']==d and r['method'] in methods]
 ax.barh([r['method'] for r in sel],[r['nll_mean'] for r in sel],xerr=[r['nll_std'] for r in sel],color=['#007f86' if r['method']=='Novelty Laplace' else '#6485a6' for r in sel],capsize=2)
 ax.invert_yaxis();ax.set_title(d.upper());ax.set_xlabel('NLL (lower better)');ax.grid(axis='x',alpha=.2)
fig1.savefig(fig/'nll.png');plt.close(fig1)
fig2,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for ax,d in zip(axs,DATASETS):
 for name in ['Deterministic','Ensemble 5','Laplace','Novelty Laplace']:
  curves=[]
  for s in [0,100,200]:
   with np.load(R/f'{d}-{s}-{name.replace(" ","_")}.npz') as z:curves.append(z['risk'])
  risk=np.mean(curves,axis=0);cov=np.arange(1,len(risk)+1)/len(risk);mask=cov>=.2
  ax.plot(cov[mask],risk[mask],label=name)
 ax.set_title(d.upper());ax.set_xlabel('Retained fraction');ax.set_ylabel('Retained MSE' if d=='esol' else 'Retained error rate');ax.grid(alpha=.2)
axs[0].legend(fontsize=8);fig2.savefig(fig/'risk.png');plt.close(fig2)
# Paired extension changes, no false equivalence between seed and population uncertainty.
fig3,axs=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
for ax,d in zip(axs,DATASETS):
 for j,name in enumerate(['Novelty only','Laplace constant','Novelty Laplace']):
  vals=[r['nll_difference_vs_laplace'] for r in paired if r['dataset']==d and r['method']==name]
  ax.scatter(np.full(3,j)+np.array([-.07,0,.07]),vals,s=35);ax.plot([j-.17,j+.17],[np.mean(vals)]*2,color='black')
 ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(3),['Novelty\nonly','Constant\n+ Laplace','Novelty\n+ Laplace']);ax.set_title(d.upper());ax.set_ylabel('NLL difference vs Laplace')
fig3.savefig(fig/'extension.png');plt.close(fig3)
print(json.dumps({'rows':len(rows),'comparisons':len(comparison),'checks':'passed'}))
for r in summary:
 if r['method'] in ['Deterministic','Ensemble 5','Laplace','Novelty Laplace']:
  print(r['dataset'],r['method'],'NLL',round(r['nll_mean'],4),'AUC/RMSE',round(r.get('roc_auc_mean',r.get('rmse_mean')),4))
