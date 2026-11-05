"""Build a concise six-page report from saved measurements, never invented scores."""
import csv,json,re,html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parent;R=ROOT/'results'
rows=list(csv.DictReader((R/'summary_metrics.csv').open()));D={(r['dataset'],r['method']):r for r in rows}
comp=list(csv.DictReader((R/'original_comparison.csv').open()));audit=json.loads((R/'split_audit.json').read_text())
frozen=json.loads((ROOT/'FROZEN.json').read_text());checks=json.loads((R/'verification.json').read_text());ev=json.loads((R/'evaluation_summary.json').read_text())
models=[json.loads(p.read_text()) for p in (ROOT/'models').rglob('summary.json')]
styles=getSampleStyleSheet()
for name,font,size,leading,color in [('Title2','Helvetica-Bold',23,27,'#17314b'),('Section2','Helvetica-Bold',15,19,'#17314b'),('Body2','Times-Roman',10.5,14,'#182635'),('Small2','Helvetica',8,10.5,'#435366'),('Eq2','Courier',9,13,'#182635')]:
 styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=8))
story=[];md=[]
def p(text,style='Body2'):
 story.append(Paragraph(text,styles[style]));md.append(re.sub('<[^>]+>','',html.unescape(text))+'\n')
def title(t):p(t,'Section2')
def page():story.append(PageBreak());md.append('\n---\n')
def table(data,widths):
 cs=ParagraphStyle('cell',fontName='Helvetica',fontSize=7.8,leading=10)
 t=Table([[Paragraph(str(v),cs) for v in row] for row in data],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#93a7b8'))]))
 story.extend([t,Spacer(1,8)]);md.extend(['| '+' | '.join(map(str,row))+' |\n' for row in data])
def fig(name,h):story.extend([Image(str(R/'figures'/name),width=493,height=h),Spacer(1,7)])
def val(d,m,k):return float(D[d,m][k+'_mean'])
def pm(d,m,k):return f'{val(d,m,k):.3f} +/- {float(D[d,m][k+"_std"]):.3f}'
def delta(d):return val(d,'Novelty Laplace','nll')-val(d,'Laplace','nll')
def stat(r,prefix):
 mean=float(r[prefix+'_mean']);sd=r.get(prefix+'_std','')
 return f'{mean:.3f} +/- {float(sd):.3f}' if sd else f'{mean:.3f} (SD n/a)'
def reference(d,m,k):return next(r for r in comp if r['dataset']==d and r['method']==m and r['metric']==k)

p('MUBen on a CPU:<br/>expanded reproduction and<br/>chemical-novelty-aware Laplace','Title2')
p('Course technical report | 4 October 2026 | 1 CPU, 4 GB RAM, no GPU','Small2')
title('1. Objective and scope')
p('MUBen benchmarks uncertainty quantification (UQ) across molecular representations and property tasks [1]. We reproduce a larger, explicitly bounded descriptor-DNN subset and test whether adding chemical unfamiliarity to last-layer Laplace improves predictive uncertainty. BACE models activity, BBBP models blood-brain-barrier permeability, and ESOL models aqueous solubility. These are screening-relevant benchmarks, not evidence of therapeutic efficacy.')
p('This expansion adds regression, Focal Loss, SWAG, five-member ensembles, source-aligned temperature fitting and checkpoint selection, an architecture sensitivity study, and controlled extension ablations. The original two-dataset study is preserved. No pretrained model is fine-tuned; conclusions about pretrained molecular representations remain untested.')
table([['Dataset','Task','Train / valid / test','Novel test count*']]+[[a['dataset'].upper(),'Regression' if a['dataset']=='esol' else 'Classification',f"{a['train_n']} / {a['valid_n']} / {a['test_n']}",a['novel_test_n']] for a in audit],[75,110,178,130])
p('*Prespecified subgroup: maximum Morgan-fingerprint similarity to training molecules below 0.4. Full supplied splits are retained; no test-driven resplitting.','Small2')
title('Measured findings')
for d in ['bace','bbbp','esol']:
 candidates=[r for r in rows if r['dataset']==d and r['method']!='Source architecture'];best=min(candidates,key=lambda r:float(r['nll_mean']))
 p(f"<b>{d.upper()}.</b> Lowest mean NLL: {best['method']} ({float(best['nll_mean']):.3f}). Ordinary Laplace: {val(d,'Laplace','nll'):.3f}; novelty-aware Laplace: {val(d,'Novelty Laplace','nll'):.3f}; paired mean difference {delta(d):+.3f} (negative favors the extension).")
p(f"The study produced {len(models)} trained models and {checks['result_rows_recomputed']} evaluated dataset/method/repeat rows. All reported metrics were recomputed from saved predictions. Three seeds quantify initialization variation on fixed test sets; they do not establish population significance. BACE/BBBP analyses are exploratory because earlier test results informed the extension hypothesis; ESOL was newly held out.")

page();title('2. Method and reproduction fidelity')
p('The main DNN follows Appendix B: 200 normalized RDKit/DescriptaStorus features, eight hidden layers of width 128, GELU and dropout 0.1. Train with AdamW (learning rate 0.0002, weight decay 0.01), batch 256, at most 400 epochs. Checkpoints accept tied validation improvements and training stops after more than 50 unsuccessful validations, matching source logic. Validation selects ROC-AUC for classification and RMSE for regression. Target mean and population SD are fitted on ESOL training labels only.')
table([['Component','Implementation and remaining deviation'],['Repeated ensembles','Three groups: seeds 0-4, 100-104, 200-204; five members per group. Original scaffold ensembles use ten [1, App.C.1].'],['Temperature / Focal Loss','Temperature: validation-only scalar, initialized at 1, AdamW lr 0.01, 20 epochs. Focal: independent training, torchvision alpha 0.25/gamma 2.'],['MC dropout / SWAG','30 samples each; SWAG: 20 extra epochs at lr 0.0001, pinned blockwise low-rank-plus-diagonal sampler. Copy snapshots to avoid CPU storage aliasing.'],['ESOL likelihood','Mean and softplus variance head, Gaussian NLL; inverse-scale mean and variance. Ensemble/MC/SWAG average variances as in the paper; no between-model term.'],['Architecture sensitivity','Separate nine-layer/ReLU deterministic variant reflects pinned defaults. Main8/GELU follows paper; published-run arguments are not reconstructed from defaults.']],[110,383])
p('Software versions, RNG initialization order and selected seed groups differ from the authors; this is not bitwise replication. FIDELITY.md records the complete matrix. The source commit is f0ec55667c1c4cc7b63369018fd011a96f31fdc4 [2].')
title('Laplace and chemical novelty')
p('Freeze the representation and fit a full 129-parameter affine mean/logit head approximation. Classification uses laplace-torch with a fixed-zero reference logit, verified against Bernoulli curvature. ESOL uses conditional Gaussian curvature with the learned noise head fixed:')
p('Sigma = (H + alpha I)^(-1);  v(x) = phi(x)^T Sigma phi(x)<br/>H_class = sum p_i(1-p_i) phi_i phi_i^T<br/>H_reg = sum phi_i phi_i^T / noise_i','Eq2')
p('Let d(x)=1-max training Tanimoto similarity, with Morgan radius 2, 2048 bits and chirality disabled. The proposed correction is v_new=v+lambda*d(x)^2. Classification predicts sigmoid[m/sqrt(1+pi*v_new/8)]; regression adds v_new to the predicted noise variance. This is a heuristic correction, not an exact Bayesian posterior.')
p('Select alpha from 10^-4 to 10^4 and lambda from {0,.01,.1,1,10,100} using validation NLL only. Compare ordinary Laplace, novelty-only variance, constant inflation plus Laplace, and novelty-aware Laplace with equal six-candidate lambda grids. Posterior centers are early-stopped estimates, not prior-specific MAP optima. ESOL models only conditional mean-head uncertainty.')

page();title('3. Comparison with the authors\' results')
p('Each row compares the paper-specified DNN implementation here with the authors\' pinned primary result CSV [2]. Values are mean +/- SD across three initialization groups here; published ensemble SD is unavailable. Classification primary metric is ROC-AUC (higher better); ESOL is RMSE (lower better). NLL is lower better. All metrics, SDs and absolute differences are in results/original_comparison.csv.')
for datasets,label in [(['bace','bbbp'],'Classification'),(['esol'],'Regression')]:
 p(label,'Small2');data=[['Dataset / method','Original primary','Our primary','Original NLL','Our NLL']]
 for d in datasets:
  for m in ['Deterministic','Temperature','MC dropout','Ensemble 5','Focal Loss','SWAG']:
   if (d,m) not in D:continue
   rr=reference(d,m,'rmse' if d=='esol' else 'roc_auc');nn=reference(d,m,'nll')
   data.append([d.upper()+' / '+m,stat(rr,'original'),stat(rr,'reproduced'),stat(nn,'original'),stat(nn,'reproduced')])
 table(data,[121,93,93,93,93])
p('These differences are descriptive, not evidence that our DNN improves on the paper. Ensemble size, architecture/source ambiguity, seeds, software and numerical details remain different. Large published BBBP NLL SD indicates unstable probability scores; close AUC does not imply close calibration. The published ensemble CSV does not provide variability across repeated ensembles.')
p('Classification comparison uses unclipped PyTorch BCE, binary Brier, trapezoidal PRC-AUC and positive-class15-bin ECE/MCE. ESOL NLL omits the Gaussian normalization constant, matching source; adding 0.91894 gives full NLL. Neither is interchangeable with classification NLL or comparable in magnitude across differently scaled regression targets.','Small2')

page();title('4. Extension results and ablations')
p('All hyperparameters were frozen before this version\'s test evaluation. The primary comparison is novelty-aware versus ordinary Laplace NLL. Constant inflation tests whether merely softening/widening predictions is sufficient; novelty-only tests whether the posterior variance is needed. Means +/- sample SD use three seeds.')
data=[['Dataset / method','NLL','Selective risk*','Selected lambda (3 seeds)']]
for d in ['bace','bbbp','esol']:
 for m in ['Laplace','Novelty only','Laplace constant','Novelty Laplace']:
  lambdas='-'
  if m!='Laplace':lambdas=', '.join(str(f['corrections'][m]['selected_lambda']) for f in frozen['fits'] if f['dataset']==d)
  data.append([d.upper()+' / '+m,pm(d,m,'nll'),pm(d,m,'aurc_mse' if d=='esol' else 'aurc'),lambdas])
table(data,[157,118,118,100])
p('*Area under retained-error curve for classification and retained-MSE curve for regression; lower better. Different task scales cannot be averaged meaningfully.','Small2')
fig('extension.png',153)
p('Figure 1. Each point is one seed\'s NLL difference from ordinary Laplace; bars show the three-seed mean. Zero means no change. These are paired initialization comparisons on the same molecules, not independent test samples.','Small2')
for d in ['bace','bbbp','esol']:
 direction='decreased' if delta(d)<-1e-8 else ('increased' if delta(d)>1e-8 else 'was unchanged')
 p(f"<b>{d.upper()}:</b> extension NLL {direction} by {abs(delta(d)):.4f}. Constant-inflation control NLL {val(d,'Laplace constant','nll'):.3f}; novelty-only {val(d,'Novelty only','nll'):.3f}. ")

p('The ablations do not establish that combining chemical novelty with Laplace is the best strategy. Novelty alone has lower NLL on both classification tasks; constant inflation slightly outperforms the combination on ESOL. The combination improves classification AURC versus Laplace, but worsens ESOL retained-MSE AURC. BACE lambda selections reach the upper grid boundary; no post-test grid expansion was made.')

page();title('5. Reliability, selection and source checks')
fig('risk.png',153)
p('Figure 2. Error among retained predictions, averaged across seeds. Classification ranks by predictive entropy; regression by predictive variance. Curves show 20-100% coverage, while saved AURC includes all nonzero coverages.','Small2')
data=[['Dataset / method','Novel count','Novel NLL','Familiar NLL']]
for d in ['bace','bbbp','esol']:
 for m in ['Laplace','Novelty Laplace']:
  sr=D[d,m]
  data.append([d.upper()+' / '+m,int(float(sr['novel_n_mean'])),f"{float(sr['novel_nll_mean']):.3f}" if sr.get('novel_nll_mean') else 'n/a',f"{float(sr['familiar_nll_mean']):.3f}" if sr.get('familiar_nll_mean') else 'n/a'])
table(data,[193,80,110,110])
p('The similarity threshold 0.4 was fixed before evaluation. The BACE novel subgroup NLL worsens slightly, despite improved overall NLL; its gain comes from more familiar molecules. BBBP and ESOL novel-subgroup NLL improve. Counts are small and these comparisons are descriptive.')
title('Regression calibration: two different quantities')
p('The pinned upstream regression_calibration_error broadcasts N labels against N predictive distributions, producing an N-by-N matrix. That is different from pairing each molecule\'s prediction with its own label. We reproduce that source value only for reference comparison and additionally report the paired probability-integral-transform (PIT) calibration error. A direct numerical check verifies exact source agreement; neither definition is silently substituted.')
table([['ESOL method','Source-compatible CE','Paired PIT CE','90%coverage / width']]+[[m,pm('esol',m,'ce_upstream'),pm('esol',m,'ce_paired'),f"{val('esol',m,'coverage90'):.3f} / {val('esol',m,'width90'):.3f}"] for m in ['Deterministic','Ensemble 5','Laplace','Novelty Laplace']],[120,125,120,128])
p('Coverage is the empirical fraction in a Gaussian 90% interval; width is in original ESOL units. A wider interval can improve coverage without improving ranking or usefulness. Paired PIT error uses 19 cumulative probabilities between 0 and 1. Source-compatible CE does not validate per-molecule calibration.','Small2')
title('Checks completed')
p(f"Recomputed all {checks['result_rows_recomputed']} result rows from stored predictions. Checked classifier curvature against analytic/autograd Hessians and library predictive probabilities; Gaussian mean-head Hessian against autograd; lambda = 0 against its base model; all checkpoint reloads; source Gaussian loss, target-scaler roundtrip and exact regression CE. Train/test canonical overlaps: "+', '.join(a['dataset'].upper()+'='+str(a['train_test_canonical_overlap']) for a in audit)+'.')

page();title('6. Compute, limitations and conclusions')
p(f"The {len(models)} saved training runs completed {sum(m['epochs'] for m in models):,} epochs in aggregate. Summed measured training invocation time was {sum(m['training_seconds'] for m in models):.1f}s. The successful fitting stage took {frozen['fit_stage_seconds']:.1f}s; evaluation took {ev['seconds']:.1f}s. Peak process RSS was {max(frozen['peak_rss_mib'],ev['peak_rss_mib']):.0f}MiB. These are measurements on one allocated CPU; process RSS excludes other job processes. Setup, downloads, report authoring and failed attempts are excluded from these stage timings.")
p('Lustre stores raw inputs, feature caches, checkpoints, optimizer/RNG restart states, uncertainty parameters, predictions, code, logs and hashes. Allocation expiry stops computation but does not delete this project directory. Institutional retention policies still apply; the downloadable archive provides a second copy. Raw inference timings are saved but differ in loading overhead, so no inference-speedup claim is made.')
title('Architecture sensitivity')
table([['Dataset','Paper 8/GELU primary','Source 9/ReLU primary','NLL: paper / source']]+[[d.upper(),pm(d,'Deterministic','rmse' if d=='esol' else 'roc_auc'),pm(d,'Source architecture','rmse' if d=='esol' else 'roc_auc'),f"{val(d,'Deterministic','nll'):.3f} / {val(d,'Source architecture','nll'):.3f}"] for d in ['bace','bbbp','esol']],[63,151,163,116])
p('Depth and activation change together, so this is sensitivity analysis rather than isolation of either effect. Source defaults cannot prove the historical configuration behind published scores. Software/RNG differences and five-versus-ten ensemble members remain limitations.')
title('What is and is not reproduced')
p('<b>Supported within scope:</b> MUBen concludes that ensembles improve on deterministic predictions [1, Sec.6]. Our five-member ensembles improve both NLL and AUC/RMSE on all three tasks. <b>Partially reproduced:</b> the paper recommends temperature scaling and MC dropout for classification; MC dropout improves NLL on both tasks here, but temperature scaling improves BACE and worsens BBBP. <b>Not tested:</b> the claims about pretrained backbones, BBP/SGLD for regression, and random versus scaffold splits. These small fixed tests cannot overturn the broader benchmark.')
p('The extension reduces NLL relative to ordinary Laplace on all three tasks, but simpler controls prevent a claim that their combination is superior. BACE/BBBP remain exploratory. Small reused validation sets, boundary hyperparameters and frozen representations constrain inference. ESOL coverage improves toward 90% while intervals widen and uncertainty ranking worsens. The contribution is a verified reproduction and controlled extension study, not a universal improvement or established research novelty.')
title('Reproduce and inspect')
p('Persistent root: /l/users/sevinch.zoda/schub/projects/muben-laplace/work/expanded_v2<br/>Run run_expansion.py --stage fit, then evaluate_saved.py, then analyze_expansion.py in an allocated CPU environment. README.md explains frozen-run protection and fresh reproduction. FROZEN.json, FIDELITY.md, original_comparison.csv and MANIFEST.json provide traceability.','Small2')
for t in ['[1] Li et al. MUBen: Benchmarking the Uncertainty of Molecular Representation Models. arXiv:2306.10060v4, 2024. https://arxiv.org/html/2306.10060v4','[2] Authors\' source and primary result CSVs, commit f0ec55667c1c4cc7b63369018fd011a96f31fdc4. https://github.com/Yinghao-Li/MUBen','[3] Daxberger et al. Laplace Redux, NeurIPS 2021; laplace-torch 0.2.3. https://github.com/AlexImmer/Laplace','[4] Related work on molecular uncertainty under shift: CoDrug, NeurIPS 2023. https://arxiv.org/abs/2310.12033']:
 p(t,'Small2')

def footer(canvas,doc):
 canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52677d'));canvas.drawString(51,28,'MUBen CPU reproduction | measured results and documented deviations');canvas.drawRightString(544,28,str(doc.page));canvas.restoreState()
SimpleDocTemplate(str(ROOT/'technical_report.pdf'),pagesize=A4,rightMargin=51,leftMargin=51,topMargin=40,bottomMargin=45,title='Expanded MUBen CPU reproduction',author='Course reproduction project').build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'technical_report.md').write_text('\n'.join(md))
print('Built technical_report.pdf')
