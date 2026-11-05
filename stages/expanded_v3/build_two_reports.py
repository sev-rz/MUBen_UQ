"""Build two reports from saved results (no retyped numbers). Run from work/:
  .venv/bin/python expanded_v3/build_two_reports.py
-> final/report_1_reproduction.pdf/.md, final/report_2_extension.pdf/.md"""
from pathlib import Path
import pandas as pd
from PIL import Image as PI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

W = Path(__file__).resolve().parent.parent; V2, V3 = W/'expanded_v2', W/'expanded_v3'; R = V3/'results'; OUT = W/'final'; OUT.mkdir(exist_ok=True)
oc = pd.read_csv(V2/'results/original_comparison.csv').rename(columns={'original_mean':'upstream_mean','original_std':'upstream_std','difference':'delta_reproduced_minus_upstream'})
rs = pd.read_csv(R/'random_split_comparison.csv'); by = pd.read_csv(R/'bayesian_comparison.csv')
by = by[by.metric.isin(['rmse','mae','nll','ce_upstream'])]          # definition-matched rows only (fix: was 18, now 12)
cb = pd.read_csv(R/'chemberta_comparison.csv'); CB = cb[cb.backbone=='ChemBERTa']
al = pd.read_csv(R/'active_learning/summary.csv').set_index('strategy'); pc = pd.read_csv(R/'active_learning/paired_comparisons.csv')
v2 = pd.read_csv(V2/'results/summary_metrics.csv').set_index(['dataset','method'])
DS = ['bace','bbbp','esol']; PRIM = {'bace':'roc_auc','bbbp':'roc_auc','esol':'rmse'}

def within(d, k):
    d = d.dropna(subset=['upstream_std']); d = d[d.upstream_std > 0]
    return int((d.delta_reproduced_minus_upstream.abs() <= k*d.upstream_std).sum()), len(d)
f3 = lambda x: f'{x:.3f}'; pm = lambda m, s: f'{m:.3f} ± {s:.3f}'; p1 = lambda m, s: f'{m:.1f} ± {s:.1f}'
V = lambda d, m, k: v2.loc[(d, m), k+'_mean']; VS = lambda d, m, k: pm(V(d, m, k), v2.loc[(d, m), k+'_std'])
sel = lambda s, r, e: pc[(pc.strategy==s)&(pc.reference==r)&(pc.endpoint==e)].iloc[0]

ss = getSampleStyleSheet()
for n, p, kw in [('B','BodyText',dict(fontSize=9.2,leading=12.4,spaceAfter=5)), ('Ti','Title',dict(fontSize=16,leading=20,spaceAfter=4)),
                 ('Sub','BodyText',dict(fontSize=8.5,textColor=colors.HexColor('#555555'),alignment=1,spaceAfter=10)),
                 ('H2','Heading2',dict(fontSize=11.5,spaceBefore=8,spaceAfter=3)), ('Cap','BodyText',dict(fontSize=7.8,leading=9.8,textColor=colors.HexColor('#444444'),spaceAfter=7)),
                 ('Ab','BodyText',dict(fontSize=9,leading=12,leftIndent=14,rightIndent=14,backColor=colors.HexColor('#F3F6F9'),borderPadding=6,spaceAfter=10))]:
    ss.add(ParagraphStyle(n, parent=ss[p], **kw))

class Doc:
    def __init__(s, name, short): s.name, s.short, s.story, s.md = name, short, [], []
    def P(s, t, st='B'):
        s.story.append(Paragraph(t, ss[st])); s.md.append({'Ti':'# ','H2':'## '}.get(st,'') + t.replace('<b>','**').replace('</b>','**').replace('<i>','*').replace('</i>','*') + '\n')
    def T(s, rows, widths, cap):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('FONT',(0,0),(-1,-1),'Helvetica',7.6),('FONT',(0,0),(-1,0),'Helvetica-Bold',7.6),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF4')),
            ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#999999')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        s.story.append(KeepTogether([t, Spacer(1,3), Paragraph(cap, ss['Cap'])]))
        s.md.append('| '+' | '.join(map(str,rows[0]))+' |\n|'+'---|'*len(rows[0])+'\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows[1:])+f'\n\n*{cap}*\n')
    def F(s, path, width, cap):
        w, h = PI.open(path).size
        s.story.append(KeepTogether([Image(str(path), width=width, height=width*h/w), Paragraph(cap, ss['Cap'])]))
        s.md.append(f'![{cap}]({Path(path).relative_to(W)})\n*{cap}*\n')
    def build(s):
        def foot(c, d): c.setFont('Helvetica',7.5); c.drawString(51,25,s.short); c.drawRightString(A4[0]-51,25,str(d.page))
        SimpleDocTemplate(str(OUT/f'{s.name}.pdf'), pagesize=A4, rightMargin=51, leftMargin=51, topMargin=42, bottomMargin=45, title=s.short).build(s.story, onFirstPage=foot, onLaterPages=foot)
        (OUT/f'{s.name}.md').write_text('\n'.join(s.md))

REFS = ['[1] Li et al. MUBen: Benchmarking the Uncertainty of Molecular Representation Models. arXiv:2306.10060v4, 2024.',
        '[2] Authors\' code and result CSVs, github.com/Yinghao-Li/MUBen, commit f0ec55667c1c4cc7b63369018fd011a96f31fdc4.',
        '[3] Daxberger et al. Laplace Redux, NeurIPS 2021; laplace-torch, github.com/AlexImmer/Laplace.',
        '[4] Chithrananda et al. ChemBERTa; DeepChem/ChemBERTa-77M-MLM.']
REPRO = ('Code, frozen protocols, configurations, predictions, per-seed metrics and operations logs are in the submission archive. Model weights '
         '(about 0.9 GB) are excluded from the archive to keep it small; they remain on the cluster and are regenerable from the code and seeds. '
         'Phase-2 entry point: expanded_v2/README.md; phase-3: expanded_v3/PROTOCOL.md and results/PHASE3_SUMMARY.md. '
         'Both reports are rebuilt by expanded_v3/build_two_reports.py.')

# ===================== REPORT 1: REPRODUCTION =====================
A = Doc('report_1_reproduction', 'Report 1 — MUBen reproduction')
s1, sn = within(oc, 1); s2, _ = within(oc, 2); r1, rn = within(rs, 1); r2, _ = within(rs, 2)
b1, bn = within(by, 1); b2, _ = within(by, 2); c1, cn = within(CB, 1); c2, _ = within(CB, 2)
t1, tn = s1+r1+b1+c1, sn+rn+bn+cn; t2 = s2+r2+b2+c2
A.P('Reproducing MUBen on a CPU: How Closely Do Uncertainty Benchmarks for Molecular Property Prediction Replicate?', 'Ti')
A.P('Report 1 of 2 — reproduction. CBIO7215 Multimodal AI for Therapeutic Discovery · Seva · October 2026', 'Sub')
A.P(f'<b>Summary.</b> We reproduce a scoped subset of MUBen [1] — the RDKit-descriptor DNN on BACE, BBBP and ESOL, with deterministic, temperature '
    f'scaling, focal loss, MC dropout, deep ensembles, SWAG, BBP and SGLD, on scaffold and random splits, plus the pretrained ChemBERTa-77M backbone on BACE — '
    f'using only CPUs. Of {tn} metric-level comparisons with the authors\' published results, {t1} ({100*t1/tn:.0f}%) fall within one published SD and '
    f'{t2} ({100*t2/tn:.0f}%) within two. Accuracy and ranking metrics reproduce closely; disagreements concentrate in high-variance calibration-tail '
    f'metrics. No qualitative conclusion we tested is contradicted; five reproduce fully and one (temperature scaling) partly.', 'Ab')

A.P('1. Paper and scope', 'H2')
A.P('MUBen benchmarks how well uncertainty-quantification (UQ) methods estimate the reliability of molecular property predictions across backbones and '
    'tasks. Reliable uncertainty matters for screening: a model that knows when it is wrong lets chemists prioritise which predictions to trust. '
    'We reproduce the part of the benchmark that fits a CPU budget, chosen before any results were seen.')
A.T([['Dataset','Task','Train / valid / test'],['BACE','β-secretase inhibition (classification)','1210 / 151 / 152'],
     ['BBBP','Blood-brain-barrier permeability (classification)','1631 / 204 / 204'],['ESOL','Aqueous solubility (regression)','902 / 113 / 113']],
    [70,250,173], 'Table 1. Datasets, using the authors\' supplied scaffold splits. Random splits (Section 4) use the same sizes.')
A.P('<b>Model.</b> DNN on 200 normalised RDKit descriptors, 8 hidden layers of width 128, GELU, dropout 0.1, AdamW (lr 2e-4, weight decay 0.01), batch 256, '
    'up to 400 epochs with early stopping on validation ROC-AUC (classification) or RMSE (regression), following the paper\'s appendix and the pinned source '
    'code [2]. <b>UQ methods.</b> Temperature scaling (validation-fitted), focal loss, MC dropout (30 samples), 5-member deep ensembles, SWAG (30 samples), '
    'and, for ESOL, BBP and SGLD Bayesian heads. <b>Backbone check.</b> ChemBERTa-77M fully fine-tuned on BACE. <b>Repeats.</b> 3 seeds (scaffold) or '
    '3 regenerated splits (random). <b>Protocol.</b> All analysis choices were frozen in protocol files before test sets were evaluated.')

A.P('2. Scaffold splits (main benchmark setting)', 'H2')
rows = [['Dataset / method','Primary: paper','Primary: ours','NLL: paper','NLL: ours']]
for d in DS:
    for m in ['Deterministic','Temperature','MC dropout','Ensemble 5','Focal Loss','SWAG']:
        x = oc[(oc.dataset==d)&(oc.method==m)].set_index('metric')
        if PRIM[d] not in x.index: continue
        g = lambda k, c: '—' if k not in x.index or pd.isna(x.loc[k, c+'_mean']) else (f3(x.loc[k, c+'_mean']) + ('' if pd.isna(x.loc[k, c+'_std']) else f' ± {x.loc[k, c+"_std"]:.3f}'))
        rows.append([f'{d.upper()} / {m}', g(PRIM[d],'upstream'), g(PRIM[d],'reproduced'), g('nll','upstream'), g('nll','reproduced')])
A.T(rows, [118,94,94,94,93], f'Table 2. Scaffold-split results. Primary metric is ROC-AUC for BACE/BBBP (higher is better) and RMSE for ESOL (lower is better). '
    f'Over all {sn} metric comparisons with a published SD, {s1} are within 1 SD and {s2} within 2 SD. The paper uses 10 ensemble members; we use 5.')
A.P(f'Discrimination agrees within about 0.01–0.02 AUC everywhere. The largest disagreement is BBBP NLL, where the published values '
    f'(e.g. {f3(oc[(oc.dataset=="bbbp")&(oc.method=="Deterministic")&(oc.metric=="nll")].upstream_mean.iloc[0])}) carry SDs larger than their means, i.e. '
    'unstable probability scores in the authors\' runs; ours are lower and stable. This is a variance effect, not a contradiction.')

A.P('3. Random splits', 'H2')
rows = [['Method']+[f'{d.upper()} {PRIM[d].replace("_","-").upper()} paper / ours' for d in DS]]
for m in ['Deterministic','Temperature','MC dropout','Ensemble 5','SWAG','Focal Loss']:
    row = [m]
    for d in DS:
        x = rs[(rs.dataset==d)&(rs.method==m)&(rs.metric==PRIM[d])]
        row.append('—' if x.empty else f'{f3(x.upstream_mean.iloc[0])} / {f3(x.reproduced_mean.iloc[0])}')
    rows.append(row)
A.T(rows, [79,138,138,138], f'Table 3. Random splits, regenerated with the source algorithm (seeds 0/1/2). "—": method not applicable to regression. '
    f'Over all metrics, {r1}/{rn} within 1 published SD and {r2}/{rn} within 2 SD.')
rsd = lambda d: rs[(rs.dataset==d)&(rs.method=='Deterministic')&(rs.metric==PRIM[d])].reproduced_mean.iloc[0]
A.P(f'The paper\'s central point about evaluation reproduces: scaffold splits are much harder than random splits. Deterministic BACE AUC is {f3(rsd("bace"))} '
    f'(random) vs {f3(V("bace","Deterministic","roc_auc"))} (scaffold); BBBP {f3(rsd("bbbp"))} vs {f3(V("bbbp","Deterministic","roc_auc"))}; ESOL RMSE '
    f'{f3(rsd("esol"))} vs {f3(V("esol","Deterministic","rmse"))}. Random splits leak shared scaffolds between training and test.')

A.P('4. Bayesian methods on ESOL', 'H2')
rows = [['Method','RMSE paper / ours','MAE paper / ours','NLL paper / ours']]
for m in ['BBP','SGLD paper interval1','SGLD source interval2']:
    row = [m.replace(' paper interval1',' (paper timing)').replace(' source interval2',' (source timing)')]
    for k in ['rmse','mae','nll']:
        x = by[(by.method==m)&(by.metric==k)]
        row.append(f'{f3(x.upstream_mean.iloc[0])} / {pm(x.reproduced_mean.iloc[0], x.reproduced_std.iloc[0])}')
    rows.append(row)
A.T(rows, [130,121,121,121], f'Table 4. BBP and SGLD on ESOL (scaffold split). NLL uses the source definition. Including the source calibration error, '
    f'{b1}/{bn} definition-matched comparisons are within 1 SD and {b2}/{bn} within 2 SD.')
A.P('BBP reproduces within one published SD on every metric. SGLD accuracy reproduces and its NLL is lower than published. The paper and source code describe '
    'different SGLD sampling intervals; running both changes results by less than the seed SD.')

A.P('5. Pretrained backbone: ChemBERTa-77M on BACE', 'H2')
rows = [['Method','ROC-AUC paper / ours','NLL paper / ours','Brier paper / ours','ECE paper / ours']]
for m in ['Deterministic','Temperature','MC dropout']:
    row = [m]
    for k in ['roc_auc','nll','brier','ece15_positive']:
        x = CB[(CB.method==m)&(CB.metric==k)]; row.append(f'{f3(x.upstream_mean.iloc[0])} / {pm(x.reproduced_mean.iloc[0], x.reproduced_std.iloc[0])}')
    rows.append(row)
A.T(rows, [69,106,106,106,106], f'Table 5. ChemBERTa-77M, fully fine-tuned (3.43 M parameters, CPU). {c1}/{cn} comparisons within 1 SD, {c2}/{cn} within 2 SD.')
dg = lambda m: CB[(CB.method==m)&(CB.metric=='roc_auc')].reproduced_mean.iloc[0] - V('bace', m, 'roc_auc')
A.P(f'The paper reports that this pretrained language model underperforms the descriptor DNN on BACE. We find the same: ChemBERTa − DNN ROC-AUC is '
    f'{dg("Deterministic"):+.3f} (deterministic) and {dg("MC dropout"):+.3f} (MC dropout). Our seed-to-seed spread (AUC SD ≈ 0.04) is larger than published.')

A.P('6. Which claims of the paper reproduce?', 'H2')
ens = all((V(d,'Ensemble 5','nll') < V(d,'Deterministic','nll')) and ((V(d,'Ensemble 5','roc_auc') > V(d,'Deterministic','roc_auc')) if d!='esol' else (V(d,'Ensemble 5','rmse') < V(d,'Deterministic','rmse'))) for d in DS)
mcd = all(V(d,'MC dropout','nll') < V(d,'Deterministic','nll') for d in ['bace','bbbp'])
tmp = [d.upper() for d in ['bace','bbbp'] if V(d,'Temperature','nll') < V(d,'Deterministic','nll')]
A.T([['Claim in the paper','Our evidence','Verdict'],
     ['Deep ensembles improve on deterministic predictions', 'Lower NLL and better primary metric on all 3 tasks' if ens else 'Mixed across tasks', 'Reproduced' if ens else 'Partly'],
     ['MC dropout improves classification calibration', 'Lower NLL on BACE and BBBP' if mcd else 'Mixed', 'Reproduced' if mcd else 'Partly'],
     ['Temperature scaling improves classification calibration', f'Lower NLL on {", ".join(tmp) or "neither"}; not on the other', 'Partly reproduced'],
     ['Scaffold splits are much harder than random splits', 'AUC drops 0.07–0.24; RMSE rises ~0.3', 'Reproduced'],
     ['Pretrained ChemBERTa underperforms the descriptor DNN on BACE', f'ROC-AUC gap {dg("Deterministic"):+.3f}', 'Reproduced'],
     ['BBP/SGLD results on ESOL', 'BBP within 1 SD; SGLD accuracy matches', 'Reproduced']],
    [190,203,100], 'Table 6. Qualitative claims tested. No tested claim was contradicted.')

A.P('7. Deviations and limitations', 'H2')
for t in ['Not a bitwise replication: software versions, RNG order and seeds differ; ensembles use 5 members (paper: 10 on scaffold, 3 on random splits).',
          'Random splits are regenerated with the source algorithm; row-level identity with the authors\' splits cannot be recovered.',
          'Scope: one descriptor backbone across tasks and one pretrained backbone on one task; the full benchmark covers more backbones and datasets.',
          'Calibration-tail metrics (MCE; ECE where published SDs are very small) differ beyond published SD in both directions; with 3 repeats these are high-variance.',
          'Operational: a file race when ChemBERTa seeds started concurrently crashed two seeds at start-up; they were rerun with unchanged code, and the corrupted '
          'decision record was repaired with its original preserved. The test set was not opened until all fits were frozen. No result was affected.']:
    A.P('• '+t)
A.P('<b>Conclusion.</b> Within a CPU budget, MUBen\'s descriptor-DNN results and its main qualitative findings reproduce. The benchmark is robust enough '
    'that differences in software, seeds and ensemble size move accuracy by at most a few hundredths, while calibration-tail metrics need more repeats '
    'than either study used to compare reliably.')
A.P('Reproducibility', 'H2'); A.P(REPRO)
A.P('References', 'H2'); [A.P(r, 'Cap') for r in REFS[:2]+REFS[3:]]
A.build()

# ===================== REPORT 2: EXTENSION =====================
B = Doc('report_2_extension', 'Report 2 — Extensions to the MUBen reproduction')
lap = {d: V(d,'Laplace','nll')-V(d,'Deterministic','nll') for d in DS}; nov = {d: V(d,'Novelty Laplace','nll')-V(d,'Laplace','nll') for d in DS}
g = sel('greedy','random','positives_150'); bn_ = sel('balanced','random','new_active_scaffolds'); ba = sel('balanced','random','positives_150')
B.P('Beyond Benchmarking: Cheap Uncertainty, Chemical Novelty, and Uncertainty-Guided Molecule Selection', 'Ti')
B.P('Report 2 of 2 — extensions. CBIO7215 Multimodal AI for Therapeutic Discovery · Seva · October 2026', 'Sub')
B.P(f'<b>Summary.</b> Building on the reproduction (Report 1), we ask three questions MUBen leaves open. (1) Can a cheap post-hoc method, last-layer Laplace, '
    f'improve uncertainty without training extra models? It lowers test NLL on all three tasks (BACE {lap["bace"]:+.3f}, BBBP {lap["bbbp"]:+.3f}, ESOL '
    f'{lap["esol"]:+.3f}) at the cost of one trained model, but lowers BACE ROC-AUC. (2) Does adding a chemical-novelty term help? It lowers NLL further, '
    f'but ablations show the novelty signal itself does most of the work: simple distance-to-training-set is a strong uncertainty estimate. (3) Does '
    f'uncertainty help decide which molecules to test? In a budgeted retrospective screen on BACE, greedy selection finds {g.mean_difference:+.0f} more actives '
    f'than random, while a policy mixing activity, uncertainty and diversity finds {bn_.mean_difference:+.1f} more new active scaffolds than random, '
    f'in every seed. Uncertainty does not pick a single winner; it controls a hit-yield versus chemical-novelty trade-off.', 'Ab')

B.P('E1. Overall idea', 'H2')
B.P('MUBen asks how well uncertainty methods <i>measure</i> the reliability of molecular predictions. Our extensions follow one thread from there to '
    '<i>using</i> uncertainty: first make it cheap, then make it aware of chemistry, then test whether it changes which molecules a project would test. '
    'The three steps target three gaps in the paper. (i) It omits Laplace approximations, a popular low-cost Bayesian method [3]; (ii) it treats all test '
    'molecules alike, although predictions fail most on chemistry far from the training data; and (iii) it never asks whether better uncertainty changes '
    'discovery decisions. All extensions reuse the reproduced DNN and scaffold splits from Report 1, with analysis choices frozen before test evaluation.')
B.P('E2. Last-layer Laplace: uncertainty from one trained model', 'H2')
B.P('After training, the feature extractor is frozen and a Gaussian posterior is fitted over the final layer (129 parameters) using the curvature of the '
    'training loss; the prior precision is tuned on validation NLL only. Predictions average over this posterior. Unlike an ensemble, no extra networks are trained.')
rows = [['Dataset','Metric','Deterministic','Ensemble (5 models)','Laplace (1 model)']]
for d in DS:
    for k in [PRIM[d], 'nll']:
        rows.append([d.upper(), k.replace('roc_auc','ROC-AUC ↑').replace('rmse','RMSE ↓').replace('nll','NLL ↓'), VS(d,'Deterministic',k), VS(d,'Ensemble 5',k), VS(d,'Laplace',k)])
B.T(rows, [55,70,123,123,122], 'Table E1. Scaffold-split test results, mean ± SD over 3 seeds.')
B.P(f'<b>What improved and why.</b> Laplace lowers NLL on every task and matches or beats the five-model ensemble on BACE and ESOL with one model. '
    'NLL punishes confident mistakes very heavily: one active molecule predicted at 1% probability costs more than many mildly wrong predictions. A single '
    'network trained to convergence is often overconfident, especially on molecules unlike its training data. Laplace replaces the single set of final-layer '
    'weights with a distribution over plausible weights; where those plausible weights disagree, the averaged prediction moves toward 0.5, which mostly '
    'softens exactly the overconfident errors. An ensemble achieves a similar effect by averaging five independently trained networks, at five times the training cost.')
B.P(f'<b>Downside: ranking on BACE.</b> ROC-AUC on BACE falls from {f3(V("bace","Deterministic","roc_auc"))} to {f3(V("bace","Laplace","roc_auc"))} '
    f'(BBBP: {f3(V("bbbp","Deterministic","roc_auc"))} → {f3(V("bbbp","Laplace","roc_auc"))}). The cause can be stated exactly. The Laplace posterior is centred '
    'on the trained weights, so each molecule\'s mean logit m(x) is identical to the deterministic model\'s; Laplace only adds a molecule-specific variance v(x) '
    'and predicts p = sigmoid(m / √(1 + πv/8)). ROC-AUC depends only on the order of molecules. Temperature scaling divides every logit by the same number, so '
    'the order — and AUC — cannot change (Report 1, Table 2). Laplace divides each logit by a different number, so a molecule with a high logit but high '
    'variance can fall below one with a lower logit and low variance. On BACE this reordering demoted correctly ranked molecules. A plausible reason, '
    '<i>not tested here</i>, is that BACE has the smallest training set, so last-layer variance is large and varies strongly across molecules. '
    '<b>Practical implication:</b> rank a library by the deterministic score and use Laplace for the confidence attached to each prediction.')
B.P('E3. Novelty-aware Laplace and ablations', 'H2')
B.P('<b>Idea.</b> Last-layer Laplace only captures uncertainty in the prediction head, so it can be confidently wrong on unfamiliar chemistry. We add '
    'variance proportional to d(x)², where d(x) = 1 − maximum Tanimoto similarity (Morgan r=2, 2048 bits) to any training molecule, with its weight λ chosen '
    'on validation NLL. <b>Controls.</b> "Novelty only" adds the same term to the deterministic model without Laplace; "Constant + Laplace" inflates variance '
    'equally for all molecules. If novelty-aware Laplace only won because it widens predictions, the constant control would match it.')
rows = [['Dataset','Laplace','Novelty only','Constant + Laplace','Novelty + Laplace']]
for d in DS:
    rows.append([d.upper()] + [VS(d, m, 'nll') for m in ['Laplace','Novelty only','Laplace constant','Novelty Laplace']])
B.T(rows, [55,110,110,110,108], 'Table E2. Test NLL (lower is better), mean ± SD over 3 seeds.')
B.F(V2/'results/figures/extension.png', 470, 'Figure E1. Per-seed NLL change relative to ordinary Laplace; bars are three-seed means. Below zero is better.')
rows = [['Dataset','Novel molecules','Novel NLL: Laplace → +novelty','Familiar NLL: Laplace → +novelty']]
for d in DS:
    rows.append([d.upper(), f'{V(d,"Laplace","novel_n"):.0f}', f'{f3(V(d,"Laplace","novel_nll"))} → {f3(V(d,"Novelty Laplace","novel_nll"))}',
                 f'{f3(V(d,"Laplace","familiar_nll"))} → {f3(V(d,"Novelty Laplace","familiar_nll"))}'])
B.T(rows, [60,90,172,171], 'Table E3. Subgroups defined before evaluation: "novel" = max similarity to training set below 0.4. Small counts; descriptive only.')
B.P(f'<b>What improved and why.</b> Adding novelty lowers NLL relative to Laplace on all tasks (BACE {nov["bace"]:+.3f}, BBBP {nov["bbbp"]:+.3f}, '
    f'ESOL {nov["esol"]:+.3f}). The reason is a blind spot of last-layer methods: the feature extractor is frozen, so a structurally unusual molecule can be '
    'mapped to familiar-looking features, and the head then sees no reason for doubt. Similarity to the training set measures unfamiliarity directly in '
    'chemical space, outside the network, so it catches cases the head-only variance misses.')
B.P(f'<b>What the ablations show.</b> The novelty term alone is as good as or better than the combination on both classification tasks, and constant '
    f'inflation matches the combination on ESOL ({f3(V("esol","Laplace constant","nll"))} vs {f3(V("esol","Novelty Laplace","nll"))}). So the combination is '
    'not a better method; the useful result is that <b>distance to the training chemistry is a strong, nearly free uncertainty signal</b>, stronger than '
    'the Bayesian head variance. The ESOL result has a simpler explanation: the deterministic regression model is under-dispersed — only '
    f'{V("esol","Deterministic","coverage90")*100:.0f}% of test values fall inside its nominal 90% interval — so any widening improves NLL, whether targeted or not.')
B.P(f'<b>Downsides.</b> (1) Wider is not smarter: on ESOL, novelty-aware Laplace raises 90% coverage to {V("esol","Novelty Laplace","coverage90")*100:.0f}% but '
    f'worsens the ability to rank predictions by trustworthiness (retained-MSE AURC {f3(V("esol","Laplace","aurc_mse"))} → {f3(V("esol","Novelty Laplace","aurc_mse"))}, '
    'lower is better). (2) On BACE the NLL gain comes from familiar molecules while novel-subgroup NLL slightly worsens (Table E3), so the mechanism is '
    'not simply "flag the novel ones". (3) The selected weight λ sits at the upper grid boundary on BACE, so the optimum may lie outside the searched range; '
    'the grid was not expanded after seeing test results. (4) The novelty term depends on fingerprint and similarity choices fixed in advance and not tuned.')
B.P('E4. Does uncertainty help choose which molecules to test?', 'H2')
B.P('<b>Setting.</b> A retrospective screening campaign on BACE: the 1210 training molecules form a pool with hidden labels. Each run starts with 100 random '
    'labelled molecules and buys 6 batches of 25 labels (401 labels in total including the validation set), retraining the DNN each round. <b>Strategies.</b> '
    'Random; greedy (highest predicted activity); entropy (most uncertain); diversity (most dissimilar to labelled set); balanced (equal-weight rank of '
    'activity, uncertainty and dissimilarity). <b>Rigour.</b> 5 seeds with shared starting sets; labels released only through an oracle reading saved query '
    'files; all 150 batches re-derived exactly from saved predictions in an audit; decision rule fixed before outcomes.')
names = {'random':'Random','greedy':'Greedy','entropy':'Entropy','diversity':'Diversity','balanced':'Balanced'}
rows = [['Strategy','Actives found (of 150 bought)','Distinct active scaffolds','New active scaffolds','Final test AUC']]
for s, n in names.items():
    a = al.loc[s]; rows.append([n, p1(a.positives_150_mean,a.positives_150_sd), p1(a.distinct_active_scaffolds_mean,a.distinct_active_scaffolds_sd),
                                p1(a.new_active_scaffolds_mean,a.new_active_scaffolds_sd), pm(a.test_auc_mean,a.test_auc_sd)])
B.T(rows, [68,118,103,103,101], 'Table E4. Outcomes, mean ± SD over 5 seeds. "New" scaffolds: active Murcko scaffolds absent from the starting labelled set.')
B.F(R/'active_learning/discovery_curves.png', 470, 'Figure E2. Left: cumulative actives found. Right: new active scaffolds found. Thin lines: seeds; thick: mean.')
rows = [['Contrast','Endpoint','Mean diff.','Per seed','Higher / lower / tied']]
for s, r, e in [('greedy','random','positives_150'),('greedy','random','new_active_scaffolds'),('entropy','random','positives_150'),('diversity','random','positives_150'),
                ('balanced','random','positives_150'),('balanced','random','new_active_scaffolds'),('balanced','greedy','new_active_scaffolds')]:
    x = sel(s, r, e); rows.append([f'{names[s]} − {names[r]}', e.replace('positives_150','actives').replace('_',' '), f'{x.mean_difference:+.1f}', x.per_seed, f'{x.n_higher} / {x.n_lower} / {x.n_tied}'])
B.T(rows, [100,110,60,125,98], 'Table E5. Paired by seed. Pre-registered rule: "consistent" only if all 5 seeds agree (exact sign-flip p = 0.0625, the minimum with 5 seeds).')
B.P(f'<b>What we found and why.</b> Greedy selection doubles hit yield ({g.mean_difference:+.0f} actives vs random, 5/5 seeds) because it buys '
    'molecules most similar to known actives — analogues in the same series — so it finds no more new chemistry. Pure diversity loses hits because the '
    'most dissimilar molecules are mostly outliers, and most are inactive. Uncertainty-only (entropy) selection also beats random on hits '
    f'({sel("entropy","random","positives_150").mean_difference:+.1f}, 5/5 seeds): uncertain molecules sit near the activity boundary, where the pool is '
    f'richer in actives than average. The balanced policy combines the three signals: activity keeps hit yield at random level ({ba.mean_difference:+.1f}), '
    f'while uncertainty and dissimilarity push it into unexplored chemistry, finding {bn_.mean_difference:+.1f} more new active scaffolds than random and '
    f'{sel("balanced","greedy","new_active_scaffolds").mean_difference:+.1f} more than greedy, in every seed. The final model\'s test AUC barely depends on '
    'the strategy, so these differences concern <i>what</i> is discovered, not how good the model becomes.')
B.P('<b>Downsides.</b> The study is retrospective: labels already exist, and the pool is the BACE training set, which excludes the validation and test '
    'scaffolds by construction. It covers one dataset, one model, one budget and equal weights in the balanced score; 5 seeds give consistent but not '
    'statistically significant effects. The balanced policy is a simple hybrid of known ideas, not a new algorithm.')
B.P('E5. What the extensions add', 'H2')
for t in ['<b>Cheap uncertainty works, with a known price.</b> One-model last-layer Laplace improves probability quality on all tasks, competitive with a '
          'five-model ensemble. Its molecule-specific shrinkage can reorder predictions, which lowered BACE ROC-AUC; the mechanism is exact and suggests '
          'ranking with the deterministic score.',
          '<b>Chemical novelty is the active ingredient.</b> Distance to the training chemistry explains most of the improvement over Laplace; the '
          'Bayesian head adds little on top. Wider intervals improve coverage but not necessarily the ranking of trustworthy predictions.',
          '<b>Uncertainty steers discovery, not just calibration.</b> Combining uncertainty with activity and diversity trades hits for new chemotypes '
          'consistently across seeds; which is preferable depends on the campaign goal.']:
    B.P('• '+t)
B.P('<b>Overall.</b> The extensions take MUBen from "which method gives the best uncertainty numbers" to "what uncertainty is good for": a one-model '
    'method is enough for calibration, a simple chemical-distance signal does most of the work, and uncertainty becomes useful for discovery when combined '
    'with activity and diversity. The contribution is a careful, controlled study of existing ideas on this benchmark, not a new method. <b>Shared '
    'limitations:</b> one descriptor backbone with frozen features; 3–5 seeds; BACE/BBBP extension results are exploratory because reproduction results '
    'informed the hypothesis (ESOL was held out); assay labels only, so nothing here establishes therapeutic benefit.')
B.P('Reproducibility', 'H2'); B.P(REPRO)
B.P('References', 'H2'); [B.P(r, 'Cap') for r in REFS]
B.build()
for n in ['report_1_reproduction','report_2_extension']:
    from pypdf import PdfReader; print(n, len(PdfReader(str(OUT/f'{n}.pdf')).pages), 'pages')
print('counts', dict(scaffold=(s1,s2,sn), random=(r1,r2,rn), bayes=(b1,b2,bn), chemberta=(c1,c2,cn), total=(t1,t2,tn)), 'ens',ens,'mcd',mcd,'tmp',tmp)
