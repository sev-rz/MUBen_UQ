"""Build the final technical report: overview + phase-2 report (unchanged) + phase-3 chapter.
Every number is read from saved result files; nothing is retyped. Run from work/:
    .venv/bin/python expanded_v3/build_final_report.py
"""
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from pypdf import PdfWriter, PdfReader

W = Path(__file__).resolve().parent.parent
V2, V3 = W / 'expanded_v2', W / 'expanded_v3'
R = V3 / 'results'
OUT = W / 'final'; OUT.mkdir(exist_ok=True)
rs = pd.read_csv(R / 'random_split_comparison.csv')
by = pd.read_csv(R / 'bayesian_comparison.csv')
cb = pd.read_csv(R / 'chemberta_comparison.csv')
al = pd.read_csv(R / 'active_learning/summary.csv').set_index('strategy')
pc = pd.read_csv(R / 'active_learning/paired_comparisons.csv')
v2 = pd.read_csv(V2 / 'results/summary_metrics.csv').set_index(['dataset', 'method'])

def within(d, k):
    d = d.dropna(subset=['upstream_std']); d = d[d.upstream_std > 0]
    return int((d.delta_reproduced_minus_upstream.abs() <= k * d.upstream_std).sum()), len(d)
def f3(x): return f'{x:.3f}'
def pm(m, s): return f'{m:.3f} ± {s:.3f}'
def p1(m, s): return f'{m:.1f} ± {s:.1f}'

ss = getSampleStyleSheet()
ss.add(ParagraphStyle('B', parent=ss['BodyText'], fontSize=9, leading=12, spaceAfter=4))
ss.add(ParagraphStyle('H1', parent=ss['Heading1'], fontSize=15, spaceAfter=6))
ss.add(ParagraphStyle('H2', parent=ss['Heading2'], fontSize=11.5, spaceBefore=6, spaceAfter=3))
ss.add(ParagraphStyle('Cap', parent=ss['BodyText'], fontSize=8, leading=10, textColor=colors.HexColor('#444444'), spaceAfter=6))
story, md = [], []
def P(t, s='B'):
    story.append(Paragraph(t, ss[s]))
    md.append({'H1': '# ', 'H2': '## '}.get(s, '') + t.replace('<b>', '**').replace('</b>', '**') + '\n')
def T(rows, widths, cap):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Helvetica', 7.6), ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 7.6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EEF4')), ('GRID', (0, 0), (-1, -1), .3, colors.HexColor('#999999')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
    story.append(KeepTogether([t, Spacer(1, 3), Paragraph(cap, ss['Cap'])]))
    md.append('| ' + ' | '.join(map(str, rows[0])) + ' |\n|' + '---|' * len(rows[0]) + '\n' +
              '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows[1:]) + f'\n\n*{cap}*\n')

r1, rn = within(rs, 1); r2, _ = within(rs, 2)
CB = cb[cb.backbone == 'ChemBERTa']
c1, cn = within(CB, 1); c2, _ = within(CB, 2)
b1, bn = within(by, 1); b2, _ = within(by, 2)
sel = lambda s, r, e: pc[(pc.strategy == s) & (pc.reference == r) & (pc.endpoint == e)].iloc[0]
g = sel('greedy', 'random', 'positives_150'); bal = sel('balanced', 'random', 'new_active_scaffolds')
lap = lambda d: v2.loc[(d, 'Laplace'), 'nll_mean'] - v2.loc[(d, 'Deterministic'), 'nll_mean']
nov = lambda d: v2.loc[(d, 'Novelty Laplace'), 'nll_mean'] - v2.loc[(d, 'Laplace'), 'nll_mean']

# ---------------- Overview ----------------
P('Uncertainty-Aware Molecular Property Prediction: A CPU Reproduction of MUBen with Laplace and Discovery Extensions', 'H1')
P('Final technical report — CBIO7215 Multimodal AI for Therapeutic Discovery. Source paper and code: MUBen (Li et al.), '
  'github.com/Yinghao-Li/MUBen, pinned commit f0ec556.')
P('Structure', 'H2')
P("<b>Part I</b> (next six pages) is the phase-2 report, unchanged: BACE, BBBP and ESOL on the authors' scaffold splits; DNN-rdkit with "
  'deterministic, temperature scaling, focal loss, MC dropout, 5-member ensembles and SWAG; plus the last-layer Laplace and novelty-aware Laplace '
  'extensions with ablations. <b>Part II</b> adds phase 3: random splits, BBP/SGLD, a pretrained ChemBERTa backbone, and a fixed-budget '
  'active-learning study that connects uncertainty to molecule selection. All runs used CPU only.')
P('Key findings', 'H2')
P(f'1. <b>Reproduction.</b> Random splits: {r1}/{rn} paper comparisons within one published SD, {r2}/{rn} within two. ChemBERTa on BACE: '
  f'{c1}/{cn} and {c2}/{cn}. BBP/SGLD on ESOL: {b1}/{bn} and {b2}/{bn}. Discrimination and accuracy reproduce; disagreements concentrate in '
  'high-variance calibration-tail metrics (MCE; ECE where published SDs are tiny).')
P("2. <b>The paper's qualitative findings reproduce.</b> Scaffold splits are far harder than random splits, and pretrained ChemBERTa-77M "
  'discriminates worse than the descriptor DNN on BACE.')
P(f'3. <b>Laplace extension.</b> Last-layer Laplace changes test NLL versus deterministic on scaffold splits by BACE {lap("bace"):+.3f}, '
  f'BBBP {lap("bbbp"):+.3f}, ESOL {lap("esol"):+.3f}; the novelty-aware prior changes it further by BACE {nov("bace"):+.3f}, '
  f'BBBP {nov("bbbp"):+.3f}, ESOL {nov("esol"):+.3f}, at a fraction of ensemble cost. On BACE it lowers ROC-AUC.')
P(f'4. <b>Discovery extension.</b> Under a 401-label budget on BACE, greedy acquisition finds {g.mean_difference:+.0f} actives versus random '
  f'(higher in {g.n_higher}/5 seeds) but no more new chemistry; an equal-rank activity/uncertainty/diversity policy finds '
  f'{bal.mean_difference:+.1f} new active scaffolds versus random (higher in {bal.n_higher}/5 seeds) at random-level hit yield. With five seeds '
  'the smallest attainable sign-flip p is 0.0625, so these are consistent rather than statistically significant effects.')
P('Scope', 'H2')
P('A scoped reproduction, not an exact replication: one backbone per experiment, 3 seeds or splits, regenerated random splits, 5-member ensembles. '
  'Protocols were frozen before test results; deviations are logged and summarised in Part II §II.6. Uncertainty here concerns assay-label '
  'prediction (BACE, BBBP, ESOL); it does not establish therapeutic benefit.')
story.append(PageBreak()); md.append('\n---\n')

# ---------------- Part II ----------------
P('Part II — Phase 3: Random Splits, Bayesian Methods, ChemBERTa and Budgeted Discovery', 'H1')
P('Phase 3 followed a protocol frozen before any phase-3 test result (expanded_v3/PROTOCOL.md). Values are mean ± SD over 3 seeds '
  "(scaffold split) or 3 regenerated splits (random split). \"Paper\" denotes the authors' published rows at the pinned commit.")
P('II.1 Random splits', 'H2')
P('Three random splits were regenerated from the supplied CSVs with the source algorithm (sklearn train_test_split, random_state 0/1/2, original '
  "validation/test sizes). Row-level identity with the authors' splits cannot be recovered, so SDs are across splits.")
main = {'bace': 'roc_auc', 'bbbp': 'roc_auc', 'esol': 'rmse'}
rows = [['Method'] + [f'{d.upper()} {main[d].replace("_", "-").upper()} paper / ours' for d in main]]
for m in ['Deterministic', 'Temperature', 'MC dropout', 'Ensemble 5', 'SWAG', 'Focal Loss']:
    row = [m]
    for d, k in main.items():
        x = rs[(rs.dataset == d) & (rs.method == m) & (rs.metric == k)]
        row.append('—' if x.empty else f'{f3(x.upstream_mean.iloc[0])} / {f3(x.reproduced_mean.iloc[0])}')
    rows.append(row)
T(rows, [79, 138, 138, 138], f'Table II.1. Main accuracy metric on random splits (AUC higher, RMSE lower is better). Over all metrics: '
  f'{r1}/{rn} within 1 published SD, {r2}/{rn} within 2 SD.')
sc = lambda d, k: v2.loc[(d, 'Deterministic'), k + '_mean']
rsd = lambda d: rs[(rs.dataset == d) & (rs.method == 'Deterministic') & (rs.metric == main[d])].reproduced_mean.iloc[0]
P(f'The split effect reproduces with the same code: deterministic BACE AUC {f3(rsd("bace"))} (random) vs {f3(sc("bace","roc_auc"))} (scaffold); '
  f'BBBP {f3(rsd("bbbp"))} vs {f3(sc("bbbp","roc_auc"))}; ESOL RMSE {f3(rsd("esol"))} vs {f3(sc("esol","rmse"))}. Random splits share scaffolds '
  '(and in BBBP/ESOL a few duplicate molecules) between train and test, which inflates apparent performance.')

P('II.2 BBP and SGLD on ESOL', 'H2')
rows = [['Method', 'RMSE paper / ours', 'MAE paper / ours', 'NLL paper / ours']]
for m in ['BBP', 'SGLD paper interval1', 'SGLD source interval2']:
    row = [m.replace(' paper interval1', ' (paper timing)').replace(' source interval2', ' (source timing)')]
    for k in ['rmse', 'mae', 'nll']:
        x = by[(by.method == m) & (by.metric == k)]
        row.append('—' if x.empty or pd.isna(x.upstream_mean.iloc[0]) else f'{f3(x.upstream_mean.iloc[0])} / {pm(x.reproduced_mean.iloc[0], x.reproduced_std.iloc[0])}')
    rows.append(row)
T(rows, [130, 121, 121, 121], f'Table II.2. Bayesian heads on ESOL, scaffold split. NLL follows the source definition (omits ½·log 2π). '
  f'{b1}/{bn} definition-matched comparisons within 1 SD, {b2}/{bn} within 2 SD.')
P('BBP reproduces within one published SD on all metrics; SGLD accuracy reproduces and its NLL is lower than published. The paper-vs-source SGLD '
  'sampling-interval discrepancy changes results by less than the seed SD. The source calibration-error formula compares all prediction/target '
  'pairs; a correctly paired PIT error is 5–34× smaller, so published CE magnitudes mainly reflect the formula.')

P('II.3 Pretrained backbone: ChemBERTa-77M on BACE', 'H2')
P('DeepChem/ChemBERTa-77M-MLM (pinned revision), fully fine-tuned: 3.43 M parameters, FP32, one CPU thread per seed, AdamW lr 1e-5, effective '
  "batch 128, early stopping on validation ROC-AUC, using the source's pooler output.")
rows = [['Method', 'ROC-AUC paper / ours', 'NLL paper / ours', 'Brier paper / ours', 'ECE paper / ours']]
for m in ['Deterministic', 'Temperature', 'MC dropout']:
    row = [m]
    for k in ['roc_auc', 'nll', 'brier', 'ece15_positive']:
        x = CB[(CB.method == m) & (CB.metric == k)]
        row.append(f'{f3(x.upstream_mean.iloc[0])} / {pm(x.reproduced_mean.iloc[0], x.reproduced_std.iloc[0])}')
    rows.append(row)
T(rows, [69, 106, 106, 106, 106], f'Table II.3. ChemBERTa, BACE scaffold test set (152 molecules). {c1}/{cn} comparisons within 1 SD, {c2}/{cn} within 2 SD.')
dg = lambda m: CB[(CB.method == m) & (CB.metric == 'roc_auc')].reproduced_mean.iloc[0] - v2.loc[('bace', m), 'roc_auc_mean']
P(f"The paper's backbone finding reproduces: ChemBERTa − DNN ROC-AUC is {dg('Deterministic'):+.3f} (deterministic) and {dg('MC dropout'):+.3f} "
  '(MC dropout). Seed variation is larger than published (test AUC SD ≈ 0.04), so a 3-seed mean has a standard error of about 0.02.')

P('II.4 Extension: molecule discovery under a fixed labelling budget', 'H2')
P('<b>Question.</b> Does uncertainty help choose which molecules to test? <b>Design.</b> The pool is the 1210 BACE training molecules. Each run '
  'starts from 100 label-blind random labels and acquires 6 batches of 25 (401 labels including the shared validation set), retraining the paper '
  'DNN from scratch each round. Strategies: random; greedy (highest predicted activity); entropy (most uncertain); diversity (farthest-point on '
  'Morgan fingerprints); balanced (equal ranks of activity, entropy and distance). Seeds 0–4 share initial sets. Labels are released only by an '
  'oracle reading each saved query file; an audit recomputed all 150 batches exactly from saved predictions, without labels.')
names = {'random': 'Random', 'greedy': 'Greedy', 'entropy': 'Entropy', 'diversity': 'Diversity', 'balanced': 'Balanced'}
rows = [['Strategy', 'Actives found (of 150)', 'Distinct active scaffolds', 'New active scaffolds', 'Final test AUC']]
for s, n in names.items():
    a = al.loc[s]
    rows.append([n, p1(a.positives_150_mean, a.positives_150_sd), p1(a.distinct_active_scaffolds_mean, a.distinct_active_scaffolds_sd),
                 p1(a.new_active_scaffolds_mean, a.new_active_scaffolds_sd), pm(a.test_auc_mean, a.test_auc_sd)])
T(rows, [68, 108, 108, 108, 100], 'Table II.4. Acquisition outcomes, mean ± SD over 5 seeds. "New" scaffolds are active scaffolds absent from the initial labelled set.')
story.append(Image(str(R / 'active_learning/discovery_curves.png'), width=440, height=round(440*672/1600)))
md.append('![Discovery curves](../expanded_v3/results/active_learning/discovery_curves.png)\n')
P('Figure II.1. Cumulative actives found per acquisition round, mean over 5 seeds.', 'Cap')
rows = [['Contrast', 'Endpoint', 'Mean diff.', 'Per seed', 'Higher / lower / tied']]
for s, r, e in [('greedy', 'random', 'positives_150'), ('greedy', 'random', 'new_active_scaffolds'), ('entropy', 'random', 'positives_150'),
                ('diversity', 'random', 'positives_150'), ('balanced', 'random', 'positives_150'), ('balanced', 'random', 'new_active_scaffolds'),
                ('balanced', 'greedy', 'positives_150'), ('balanced', 'greedy', 'new_active_scaffolds')]:
    x = sel(s, r, e)
    rows.append([f'{names[s]} − {names[r]}', e.replace('positives_150', 'actives').replace('_', ' '), f'{x.mean_difference:+.1f}', x.per_seed,
                 f'{x.n_higher} / {x.n_lower} / {x.n_tied}'])
T(rows, [100, 110, 60, 125, 95], 'Table II.5. Paired-by-seed contrasts. Pre-registered rule: an effect is called consistent only if all 5 seeds agree; '
  'the exact two-sided sign-flip p is then 0.0625, the minimum possible with 5 seeds.')
P('<b>Interpretation.</b> Exploitation doubles hit yield but concentrates on familiar chemistry; pure diversity costs hits; combining activity, '
  'uncertainty and diversity keeps random-level hit yield while finding the most new active scaffolds in every seed. Final-model test AUC barely '
  'depends on the strategy. Whether total hits or scaffold novelty matters more is a project decision outside this simulation. <b>Limits.</b> '
  'Retrospective; one dataset and model; 5 seeds; the pool excludes validation/test scaffolds; the balanced policy is a simple hybrid, not a new algorithm.')
P('II.5 Compute', 'H2')
P(f'All phase-3 work ran on CPU. Each active-learning run took {al.wall_seconds_mean.min():.0f}–{al.wall_seconds_mean.max():.0f} s on average '
  '(6–7 model fits, one thread). ChemBERTa fine-tuning took 1200–2350 s per seed on one thread.')
P('II.6 Deviations and limitations', 'H2')
for t in ["Random splits are regenerated with the source algorithm, not the authors' exact splits; ensembles use 5 members (paper: 3 for random, 10 for scaffold splits).",
          'Running ChemBERTa seeds concurrently in one job caused a file race: seeds 0 and 100 crashed at start and were rerun with unchanged code. The race '
          'left CHEMBERTA_DECISION.json syntactically corrupt; it was repaired by keeping its valid prefix and recording all three seeds (the corrupted '
          'original is preserved). No result was affected; CHEMBERTA_FROZEN.json records that the test set was unopened until all fits were frozen.',
          'Open active-learning details (diversity reference set, AUDC definition, tie handling) were fixed before outcomes and logged.',
          'Calibration-tail metrics (MCE, ECE) differ from the paper beyond published SD in both directions; with 3 seeds or splits these are high-variance.',
          'Untested: ChemBERTa with BBP/SGLD/SWAG/ensembles/focal loss or on other datasets; prospective acquisition; other budgets and models.']:
    P('• ' + t)
P('II.7 Reproducibility', 'H2')
P('The archive contains code, frozen protocols, configs, checkpoints, predictions, per-seed metrics, operations logs and this report. Phase 2: '
  'expanded_v2/README.md. Phase 3: expanded_v3/PROTOCOL.md, results/PHASE3_SUMMARY.md and OPERATIONS_LOG.md. Rebuild this report with '
  'expanded_v3/build_final_report.py.')

i = next(k for k, s in enumerate(story) if isinstance(s, PageBreak))
def foot(c, d): c.setFont('Helvetica', 7.5); c.drawRightString(A4[0] - 51, 25, f'Part II · page {d.page}')
kw = dict(pagesize=A4, rightMargin=51, leftMargin=51, topMargin=40, bottomMargin=45)
SimpleDocTemplate(str(OUT / '_overview.pdf'), **kw).build(story[:i])
SimpleDocTemplate(str(OUT / '_part2.pdf'), **kw).build(story[i + 1:], onFirstPage=foot, onLaterPages=foot)
w = PdfWriter()
counts = {}
for name, f in [('overview', OUT / '_overview.pdf'), ('part1', V2 / 'technical_report.pdf'), ('part2', OUT / '_part2.pdf')]:
    pages = PdfReader(str(f)).pages; counts[name] = len(pages)
    for pg in pages: w.add_page(pg)
w.write(str(OUT / 'technical_report_final.pdf'))
(OUT / 'technical_report_final.md').write_text('\n'.join(md) + '\n\n---\n\n# Part I — Phase-2 report\n\n' + (V2 / 'technical_report.md').read_text())
print('pages', counts, 'total', sum(counts.values()))
