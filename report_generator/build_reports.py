#!/usr/bin/env python3
"""Builds Report 1 (reproduction) and Report 2 (extensions) as print-ready HTML (A4); Chrome prints them to PDF.
All numbers are copied from recorded cells and notes of the sc-hub project muben-laplace (journal references in the appendices)."""
import html, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = {'DNN': '#0072B2', 'ChemBERTa': '#E69F00', 'GIN': '#009E73', 'a': '#0072B2', 'b': '#D55E00'}


def esc(s):
    return html.escape(str(s))


CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #1b1f24; }
h1 { font-size: 21pt; line-height: 1.15; margin: 0 0 4pt 0; color: #0b2545; }
.sub { font-size: 11pt; color: #44505c; margin-bottom: 10pt; }
.meta { font-size: 8.8pt; color: #5b6772; border-top: 2px solid #0b2545; padding-top: 5pt; margin-bottom: 12pt; }
h2 { font-size: 13.5pt; color: #0b2545; border-bottom: 1.2px solid #c9d3df; padding-bottom: 2pt; margin: 17pt 0 6pt 0; page-break-after: avoid; }
h3 { font-size: 11pt; color: #1d3557; margin: 11pt 0 3pt 0; page-break-after: avoid; }
p { margin: 0 0 6pt 0; text-align: left; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; }
li { margin-bottom: 2.5pt; }
.box { border: 1px solid #b8c7d9; background: #f2f6fb; border-radius: 4px; padding: 8pt 10pt; margin: 6pt 0 10pt 0; page-break-inside: avoid; }
.box h3 { margin-top: 0; }
.warn { border-color: #e0b36a; background: #fff8ec; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 3pt 0; font-size: 8.6pt; page-break-inside: avoid; }
th { background: #0b2545; color: #fff; font-weight: 600; text-align: left; padding: 3pt 5pt; }
td { padding: 2.6pt 5pt; border-bottom: 0.6px solid #d8e0e8; vertical-align: top; }
tr:nth-child(even) td { background: #f6f8fb; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.cap { font-size: 8.6pt; color: #4b5864; margin: 2pt 0 9pt 0; }
.cap b { color: #1d3557; }
figure { margin: 6pt 0 8pt 0; page-break-inside: avoid; }
figure img, figure svg { width: 100%; height: auto; }
.small { font-size: 8.8pt; color: #4b5864; }
.pill { display: inline-block; padding: 0 5pt; border-radius: 8px; font-size: 8pt; color: #fff; }
.ok { background: #2a9d8f; } .bad { background: #c0392b; } .mid { background: #8a8f98; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #eef1f5; padding: 0 2px; border-radius: 2px; }
.pb { page-break-before: always; }
footer { margin-top: 14pt; border-top: 1px solid #c9d3df; padding-top: 4pt; font-size: 8pt; color: #6b7682; }
"""


def table(header, rows, num_cols=(), widths=None, cap=None):
    out = ['<table>']
    if widths:
        out.append('<colgroup>' + ''.join(f'<col style="width:{w}">' for w in widths) + '</colgroup>')
    # cells are authored markup (entities such as &minus; and inline tags are intended), so they are not escaped again
    out.append('<tr>' + ''.join(f'<th class="{"n" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(header)) + '</tr>')
    for r in rows:
        out.append('<tr>' + ''.join(f'<td class="{"n" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + '</tr>')
    out.append('</table>')
    if cap:
        out.append(f'<div class="cap">{cap}</div>')
    return '\n'.join(out)


def grouped_bars(title, cats, series, colors, ymin, ymax, ytitle='', w=760, h=250, fmt='{:+.2f}', zero_line=True, label_vals=False, ticks=None):
    """series: list of (name, values). Returns inline SVG."""
    ml, mr, mt, mb = 52, 12, 26, 58
    pw, ph = w - ml - mr, h - mt - mb
    n, k = len(cats), len(series)
    gw = pw / n
    bw = gw * 0.78 / k

    def y(v):
        return mt + ph * (1 - (v - ymin) / (ymax - ymin))

    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif" font-size="9">']
    s.append(f'<text x="{ml}" y="14" font-size="10.5" font-weight="600" fill="#1d3557">{esc(title)}</text>')
    tk = ticks or [ymin + (ymax - ymin) * i / 4 for i in range(5)]
    for t in tk:
        s.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e3e8ee"/>')
        s.append(f'<text x="{ml-5}" y="{y(t)+3:.1f}" text-anchor="end" fill="#55606c">{fmt.format(t) if abs(t) < 100 else int(t)}</text>')
    if zero_line and ymin < 0 < ymax:
        s.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="#333" stroke-width="0.9"/>')
    for i, c in enumerate(cats):
        gx = ml + i * gw
        for j, (name, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            x0 = gx + gw * 0.11 + j * bw
            y0, y1 = y(max(v, 0) if ymin < 0 else v), y(min(v, 0) if ymin < 0 else ymin)
            if ymin >= 0:
                y0, y1 = y(v), y(ymin)
            vv = max(min(v, ymax), ymin)
            yy = y(vv)
            top, bot = (min(y(0), yy), max(y(0), yy)) if ymin < 0 else (yy, y(ymin))
            s.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{bw*0.92:.1f}" height="{max(bot-top,0.6):.1f}" fill="{colors[j]}"/>')
            if label_vals:
                ty = top - 2 if (v >= 0 or ymin >= 0) else bot + 9
                s.append(f'<text x="{x0+bw*0.46:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="7.6" fill="#333">{fmt.format(v)}</text>')
        s.append(f'<text x="{gx+gw/2:.1f}" y="{h-mb+14}" text-anchor="middle" fill="#333">{esc(c)}</text>')
    lx = ml
    for j, (name, _) in enumerate(series):
        s.append(f'<rect x="{lx}" y="{h-16}" width="9" height="9" fill="{colors[j]}"/><text x="{lx+13}" y="{h-8}" fill="#333">{esc(name)}</text>')
        lx += 22 + 6.2 * len(name)
    if ytitle:
        s.append(f'<text x="11" y="{mt+ph/2}" transform="rotate(-90 11 {mt+ph/2})" text-anchor="middle" fill="#55606c">{esc(ytitle)}</text>')
    s.append('</svg>')
    return '\n'.join(s)


def page(title, sub, meta, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title><style>{CSS}</style></head><body>
<h1>{esc(title)}</h1><div class="sub">{sub}</div><div class="meta">{meta}</div>
{body}
<footer>Project <code>muben-laplace</code> (sc-hub, MBZUAI Slurm cluster). Every number in this document comes from a recorded journal cell or note; references are listed in the appendix. Generated 14 October 2026.</footer>
</body></html>"""


# ======================================================================================================================
# REPORT 1
# ======================================================================================================================


def code_section(n):
    """Shared 'Code, data and reproducibility' section (both reports)."""
    out = []
    out.append('<h2 class="pb">Code, data and reproducibility</h2>')
    out.append("""<p>The code, the frozen protocols (with sha256 digests), the per-seed metric files and the summary tables behind every table in this report are in the accompanying package <code>muben_reports_code_package.zip</code> (README, <code>CODE_INDEX.md</code> with every script and its purpose, and <code>MANIFEST.csv</code> with a sha256 for every file). Each stage of the work is one folder; each folder has its <code>PROTOCOL.md</code>, written and hashed before the runs it governs, and the scripts that ran them. Every number in the text comes from a recorded cell or note of the sc-hub project <code>muben-laplace</code> (references in the appendix).</p>""")
    rows = [
        ['expanded_v4', 'Tier-B pipeline (DNN, ChemBERTa): training units, evaluation behind FROZEN markers, comparison with the published tables', 'Report 1, sections 1 to 4'],
        ['expanded_v8_gin, expanded_v10_unimol', 'GIN and Uni-Mol with the authors\' encoders, their Laplace stages and controls, bridge checks', 'Report 1, 5 to 5b; Report 2'],
        ['expanded_v9_checks', 'bridge analysis, descriptor drift, random-split identity, statistics (check E), shift-aware conformal (check F)', 'Report 1, 6; Report 2, 7'],
        ['expanded_v11, v12, v17', 'ChemBERTa schedule and head-initialisation arms; GIN with the linear schedule', 'Report 1, 6.5 and 6.6'],
        ['expanded_v6_laplace, v13, v14, v15, v18', 'last-layer Laplace, schedule test, ten or more networks, recalibration baseline and wide grid, evidence-based prior, Laplace on every ensemble member', 'Report 2, 1 to 4, 8.1, 8.2, 9'],
        ['expanded_v5_conformal, v16_round1', 'split conformal, few-label and weighted conformal, conformal under the linear schedule', 'Report 2, 5, 8.3, 8.4'],
    ]
    out.append(table(['Stage folder(s)', 'Content', 'Used in'], rows, widths=['24%', '52%', '24%']))
    out.append("""<ul><li><b>Run order.</b> Reproduction: <code>prep</code> (features), <code>worker</code> (resumable training units), <code>freeze</code> (FROZEN marker per split and dataset with the protocol and code digests), <code>evaluate</code> (test metrics only behind a FROZEN marker); then the comparison scripts, the bridge checks, Laplace (<code>checks</code>, <code>fit</code>, <code>novelty</code>, <code>evaluate</code>; the analytic Hessians are checked against automatic differentiation at every start), the conformal scripts (validation predictions from saved networks first) and the follow-up checks.</li>
<li><b>Inputs not in the package.</b> The authors' code (MUBen, github.com/Yinghao-Li/MUBen, commit f0ec556; imported unmodified), its data files and published tables, the public Uni-Mol and ChemBERTa-77M-MLM checkpoints, trained checkpoints, cached features and per-molecule prediction files (regenerated by the scripts). Several scripts contain absolute cluster paths and need the path line at the top edited to run elsewhere.</li>
<li><b>Environment.</b> Locked requirement files are included (CPU base, GPU, Laplace); runs used deterministic PyTorch algorithms in FP32 on one GPU of the MBZUAI Slurm cluster (partitions ws-ia and gpu).</li>
<li><b>Rigor.</b> Settings, including the Laplace prior strength and novelty weight, were chosen on validation data only and frozen before test predictions; protocols and decision rules were registered with hashes before each analysis; the test sets had been used for the baselines, so all extension results are labelled exploratory. Marginal deviations are stated where they occur (for example the DNN Laplace reproduction differs from the earlier per-seed numbers by up to 2.05&times;10<sup>&minus;6</sup> against a registered tolerance of 10<sup>&minus;6</sup>).</li></ul>""")
    return out

def report1():
    b = []
    b.append("""<div class="box"><h3>Summary</h3>
<p><b>Question.</b> How closely do the published MUBen results (Li et al., arXiv 2306.10060v4; code commit f0ec556) reproduce on the eight core datasets for four backbones (DNN, ChemBERTa and GIN in full; Uni-Mol on the scaffold split with four methods), and where they do not, is the cause our implementation or the published run?</p>
<p><b>Answer: partly, and it depends on split and metric.</b></p>
<ul>
<li><b>Agreement counts.</b> Under a frozen protocol we reproduced the DNN and ChemBERTa (336 fits each; scaffold and random splits; eight UQ methods): of 1,072 rows with a published standard deviation, <b>573 (53.5 %) agree within 1 SD and 828 (77.2 %) within 2 SD</b>. GIN (2D graph; scaffold only, as published) agrees in 86 of 236 rows within 1 SD and 166 within 2 SD. The paper's method claims mostly reproduce in direction.</li>
<li><b>The counts overstate the disagreement.</b> The published SDs come from three seeds and are tiny. In natural units the Deterministic GIN differs from the published value by a median of 2.8 % in ROC-AUC and 6.0 % in RMSE; the most extreme z-score (QM7 RMSE, z = &minus;21.7) is a 1.2 % difference measured against a published SD of 0.06 % of the value.</li>
<li><b>The largest gaps</b> are regression NLL and calibration error, ChemBERTa on scaffold splits, and DNN on QM7. The authors' own code, run with the paper's settings, reproduces our DNN QM7 numbers (NLL 5.47 against our 5.48) and not the published ones (10.30); the same holds for ChemBERTa on ESOL (published NLL 2.28; authors' trainer 0.76; ours 1.03). So these gaps are not caused by our reimplementation. Their cause stays open because the authors release only aggregated tables, not predictions.</li>
<li><b>Bridge checks against the authors' trainers.</b> With 10 seeds, GIN regression is consistent in 16 of 16 comparisons (the earlier ESOL miss of 2.07 SD was a three-seed artefact); GIN classification was consistent in 14 of 20, with five differences in which our GIN was better (head initialisation ruled out); the cause is a learning-rate schedule: the authors' GIN code defaults to a linear schedule, and with that schedule our GIN is consistent with their runs in <b>20 of 20</b> comparisons (section 6.6). ChemBERTa was consistent in only 4 of 8 cells, but the differences came from a learning-rate schedule we had set in the authors' trainer: with matching schedules the two pipelines agree (ESOL NLL 1.037 against 1.033 with the constant default; 0.764 against 0.758 with a linear schedule).</li>
<li><b>Uni-Mol (3D).</b> With the authors' encoder and data classes inside our pipeline (scaffold split; Deterministic, MC dropout, temperature scaling, SWAG, BBP, SGLD, focal loss and ensembles): <b>129 of 248 rows within 1 SD (52.0 %), 188 within 2 SD (75.8 %)</b>, in line with the DNN and ChemBERTa totals. Agreement depends on the method: MC dropout 70 % and Deterministic 53 % within 1 SD, SWAG and SGLD 38 %. (A first reading over four methods only gave 58.7 % and 89.4 %; it overstated the agreement and is superseded.) The authors' unmodified trainer agrees with our Deterministic Uni-Mol on <b>13 of 13</b> comparisons, so the pipeline is interchangeable with theirs.</li>
<li><b>Cross-backbone paper claims</b> (larger models are worse calibrated; the UQ method matters more for larger models) were registered before any GIN result was read and are <b>not supported</b>: at most 4 of 8 datasets, the rule needs 6.</li>
<li><b>Audit.</b> The DNN descriptor-version difference (RDKit 2024.09 against the authors' 2023.09) is immaterial. Our random splits are <b>verified identical</b> to the authors' routine applied to the Uni-Mol source files (all 8 datasets, random_state 0 to 2), so random-split differences from the published tables are not due to different test molecules.</li>
</ul>
<p><b>Added on 14 October: Laplace against a one-parameter recalibration, a wider grid for all backbones, and what repairs conformal classification.</b> (i) <b>A single fitted scalar per task (a variance multiplier in regression, a temperature in classification, fitted on validation data) reproduces or exceeds most of the Laplace gain</b>: with the registered rule, Laplace is better than the recalibration on 0 of 8 datasets for the DNN, 1 for ChemBERTa, 2 for GIN and 0 for Uni-Mol, and worse on 5, 3, 0 and 2; so much of the gain is generic variance inflation (section 8.1). (ii) A wider prior grid (10<sup>&minus;8</sup> to 10<sup>4</sup>) leaves the Laplace verdict unchanged on 8 of 8 datasets for every backbone (section 8.2). (iii) Conformal classification coverage on scaffold, 0.75 to 0.82, is restored to 0.89 to 0.90 by about 20 labelled molecules from the target scaffolds used alone (sets 16 to 33 % larger), more slowly when they are pooled with the validation scores, and partly (0.86 to 0.91, with 35 to 42 % larger sets, 16 to 24 % of them the trivial full label set) by similarity weighting that needs no labels (section 8.3). (iv) The coverage failure is the same under the linear learning-rate schedule (0.82 for Deterministic and Laplace), although the Laplace gain is not (section 8.4).</p>
<p><b>Round 2 (section 9).</b> Choosing the prior strength by the marginal likelihood (no validation data) never hits a grid edge and still improves NLL on most datasets (7, 7, 4, 6 of 8), but gives a much smaller gain than the validation-tuned prior (mean &minus;0.009 to &minus;0.109 against &minus;0.077 to &minus;0.221). Applying Laplace to every member of the ten-network ensemble still improves the ensemble NLL for the DNN, ChemBERTa and Uni-Mol (4 to 5 of 8 datasets; gains 30 to 57 % smaller than for a single network) but not for the GIN, and recalibrating each member with one scalar is as good or better on three of four backbones.</p>
<p><b>How sure.</b> Solid for the agreement counts and the metric recomputation. Weaker for the authors'-code checks, which cover Deterministic only, scaffold only, and a few datasets, with three to ten seeds. Everything rests on three seeds or splits per cell.</p></div>""")

    b.append('<h2>Why this matters for therapeutic discovery</h2>')
    b.append("""<p>Computational drug discovery screens large numbers of compounds before any is made or tested. Property models guide that choice, so they matter most for <b>new compounds that do not resemble the training set</b>, and a model that states how uncertain it is lets a chemist decide which predictions to trust. MUBen asks exactly that: how reliable are the uncertainty estimates of molecular property models? This study checks how far the published answers can be reproduced and extends them.</p>
<ul><li><b>Datasets with a therapeutic meaning.</b> BBBP (blood-brain-barrier penetration), BACE (BACE-1 inhibitor activity), Tox21 (12 toxicity assays) and ClinTox (failure of clinical trials because of toxicity, and approval status) are safety and activity endpoints; ESOL (solubility), FreeSolv (hydration free energy) and Lipo (lipophilicity) are physicochemical properties that drive absorption and distribution. QM7 (a quantum-chemistry property) is a benchmark included for breadth and has no therapeutic meaning.</li>
<li><b>Scaffold splits are the &ldquo;unseen drug&rdquo; setting.</b> A scaffold split puts compounds with a core structure that never occurs in the training set into the test set, which mimics screening new chemical series. Random splits mimic the easier case in which close analogues of the training compounds are tested. Every result is reported for both, and the main findings differ between them: models are overconfident on scaffold splits, and conformal classification sets lose coverage there.</li>
<li><b>Four views of a molecule.</b> The four backbones are four representations of the same compound: a vector of 200 computed descriptors (DNN), the SMILES string read by a pre-trained chemical language model (ChemBERTa), the molecular graph (GIN) and eleven 3D conformers per molecule (Uni-Mol, scaffold split only). They are compared on identical data and splits; they are not fused into one model, so this study compares representations and does not integrate modalities.</li></ul>""")

    b.append('<h2>1. What was reproduced and how</h2>')
    b.append("""<p>The study reproduces the MUBen benchmark for three backbones on the eight core datasets, with the authors' scaffold splits and with random splits, and the eight UQ methods of the paper. The pipeline is our own implementation, extended from earlier versions of this project. It imports the authors' SWAG, BBP and SGLD modules and, for GIN, the authors' GIN encoder and graph featuriser, and it copies the authors' metric definitions after reading their source. Everything was frozen before results were opened: a protocol with its hash, a FROZEN marker per backbone, split and dataset, and test evaluation only behind the markers.</p>""")
    b.append(table(['Item', 'Setting'], [
        ['Backbones', 'DNN on 200 normalised RDKit descriptors (8 linear 128-wide GELU layers, dropout 0.1); ChemBERTa-77M-MLM (pinned checkpoint, GPU, FP32); GIN (authors\' encoder and featuriser; 5 layers, 128 hidden units, 0.26 M parameters)'],
        ['Datasets', 'Classification: BACE, BBBP, ClinTox (2 tasks), Tox21 (12 tasks, missing labels). Regression: ESOL, FreeSolv, Lipo, QM7'],
        ['Splits', 'Scaffold: the authors\' files. Random: three splits generated with the authors\' algorithm (two successive train_test_split calls, random_state = split index, original sizes). Published tables contain no GIN random-split rows, so GIN on random splits is an extension without a target'],
        ['Methods', 'Deterministic, temperature scaling and focal loss (classification only), MC dropout, SWAG, BBP, SGLD, deep ensembles'],
        ['Seeds', 'Scaffold: 0, 100, 200 for single-model methods; one pooled 10-member ensemble (as published, no SD). Random: one seed per split, 3-member ensembles'],
        ['Agreement rule', 'A row agrees at k if |reproduced mean &minus; published mean| &le; k &times; published SD; counted only over rows with a published SD'],
    ], widths=['17%', '83%']))

    b.append('<h2>2. Results against the published tables</h2>')
    b.append(table(['Backbone', 'Split', 'Rows', 'Within 1 SD', 'Within 2 SD'], [
        ['DNN', 'scaffold', 248, '114 (46.0 %)', '185 (74.6 %)'],
        ['DNN', 'random', 288, '166 (57.6 %)', '227 (78.8 %)'],
        ['ChemBERTa', 'scaffold', 248, '93 (37.5 %)', '162 (65.3 %)'],
        ['ChemBERTa', 'random', 288, '200 (69.4 %)', '254 (88.2 %)'],
        ['GIN', 'scaffold', 236, '86 (36.4 %)', '166 (70.3 %)'],
        ['<b>DNN + ChemBERTa</b>', '', '<b>1,072</b>', '<b>573 (53.5 %)</b>', '<b>828 (77.2 %)</b>'],
    ], num_cols=(2,), cap='<b>Table 1.</b> Agreement with the published tables (journal cells c0242, c0248). GIN has 236 countable rows because 12 ClinTox rows have a published SD of exactly 0.'))
    svg = grouped_bars('Share of rows within k published SD', ['DNN scaffold', 'DNN random', 'ChemBERTa scaffold', 'ChemBERTa random', 'GIN scaffold'],
                       [('within 1 SD', [46.0, 57.6, 37.5, 69.4, 36.4]), ('within 2 SD', [74.6, 78.8, 65.3, 88.2, 70.3])], [C['a'], '#56B4E9'], 0, 100, '% of rows', fmt='{:.0f}', zero_line=False, label_vals=True, ticks=[0, 25, 50, 75, 100])
    b.append(f'<figure>{svg}<div class="cap"><b>Figure 1.</b> Agreement by backbone and split. ChemBERTa agrees best on random splits and worst on scaffold splits.</div></figure>')
    b.append(table(['Dataset', 'DNN scaffold', 'DNN random', 'ChemBERTa scaffold', 'ChemBERTa random', 'GIN scaffold'], [
        ['BACE', '27 / 34 of 42', '41 / 44 of 48', '25 / 34 of 42', '39 / 43 of 48', '16 / 31 of 42'],
        ['BBBP', '27 / 40 of 42', '37 / 43 of 48', '17 / 29 of 42', '39 / 46 of 48', '11 / 29 of 42'],
        ['ClinTox', '13 / 32 of 42', '33 / 45 of 48', '18 / 24 of 42', '35 / 42 of 48', '12 / 20 of 30'],
        ['Tox21', '18 / 30 of 42', '12 / 28 of 48', '15 / 32 of 42', '37 / 42 of 48', '14 / 27 of 42'],
        ['ESOL', '13 / 18 of 20', '11 / 21 of 24', '3 / 5 of 20', '13 / 21 of 24', '9 / 17 of 20'],
        ['FreeSolv', '5 / 11 of 20', '7 / 16 of 24', '7 / 12 of 20', '12 / 20 of 24', '12 / 19 of 20'],
        ['Lipo', '11 / 16 of 20', '14 / 16 of 24', '2 / 8 of 20', '10 / 21 of 24', '9 / 17 of 20'],
        ['QM7', '0 / 4 of 20', '11 / 14 of 24', '6 / 18 of 20', '15 / 19 of 24', '3 / 6 of 20'],
    ], cap='<b>Table 2.</b> Rows within 1 SD / within 2 SD, of the rows with a published SD, per dataset (journal cells c0130, c0248).'))
    b.append(table(['Claim (paper)', 'DNN scaffold', 'DNN random', 'ChemBERTa scaffold', 'ChemBERTa random', 'GIN scaffold'], [
        ['C1 Ensembles best on RMSE / ROC-AUC (of 8)', '6 &rarr; 7', '4 &rarr; 6', '3 &rarr; 2', '4 &rarr; 5', '7 &rarr; 5'],
        ['C2 MC dropout NLL &lt; Deterministic (of 8)', '6 &rarr; 7', '7 &rarr; 7', '8 &rarr; 6', '8 &rarr; 8', '3 &rarr; 6'],
        ['C2 Temperature NLL &lt; Deterministic (of 4)', '4 &rarr; 3', '4 &rarr; 4', '4 &rarr; 4', '4 &rarr; 4', '3 &rarr; 3'],
        ['C3 BBP ROC-AUC &lt; Deterministic (of 4)', '2 &rarr; 4', '3 &rarr; 3', '3 &rarr; 4', '3 &rarr; 4', '4 &rarr; 3'],
        ['C3 SGLD ROC-AUC &lt; Deterministic (of 4)', '2 &rarr; 1', '3 &rarr; 4', '2 &rarr; 2', '3 &rarr; 2', '4 &rarr; 3'],
    ], cap='<b>Table 3.</b> Datasets on which each paper claim holds, published &rarr; reproduced. The directions mostly reproduce; the largest changes are for GIN (C2 MC dropout, C1).'))
    b.append(table(['Metric (median z = (reproduced &minus; published) / published SD)', 'DNN scaffold', 'DNN random', 'ChemBERTa scaffold', 'ChemBERTa random'], [
        ['Classification ROC-AUC', '0.60', '&minus;0.09', '<b>&minus;2.05</b>', '0.01'],
        ['Regression RMSE', '0.08', '&minus;0.39', '<b>1.74</b>', '0.52'],
        ['Regression NLL', '&minus;0.84', '&minus;1.40', '&minus;1.66', '&minus;0.34'],
        ['Regression CE', '&minus;0.62', '&minus;0.71', '<b>&minus;3.04</b>', '&minus;1.98'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 4.</b> Where the disagreement is concentrated (journal cell c0131). Negative z means the reproduced value is lower than published.'))
    b.append("""<p class="small">Reading guide. A difference of 0.01 to 0.02 ROC-AUC or a few percent of RMSE can be several published SDs, so the counts measure agreement relative to the published seed noise, not closeness in natural units. An earlier summary of ours quoted a range of differences from one backbone's rows only; it was corrected after computing the range over every cell (note n0047).</p>""")

    b.append('<h2>3. The regression NLL and calibration gap</h2>')
    b.append("""<p>The reproduced regression NLL is in several cells far below the published one. Three descriptive checks on saved test predictions, without retraining, narrow the cause.</p>
<ul><li><b>Not an arithmetic or definition bug.</b> An independent numpy recomputation of the Gaussian NLL (source definition) equals the stored NLL for all 272 regression groups (largest difference 0.0).</li>
<li><b>Not one constant convention.</b> The gap (reproduced minus published, Deterministic) runs from +0.42 (DNN scaffold FreeSolv) to &minus;4.82 (DNN scaffold QM7).</li>
<li><b>Calibration.</b> Our learned variances are close to calibrated on random splits and overconfident on scaffold splits (Table 6). The absolute CE gap is small (about 0.007 on values of 0.03 to 0.05); the large CE z-scores come from tiny published SDs.</li></ul>""")
    b.append(table(['Backbone', 'Dataset', 'Scaffold: published', 'Scaffold: reproduced', 'Random: published', 'Random: reproduced'], [
        ['ChemBERTa', 'ESOL', '2.280', '1.033', '0.377', '0.258'],
        ['ChemBERTa', 'FreeSolv', '1.786', '1.651', '0.827', '0.784'],
        ['ChemBERTa', 'Lipo', '1.237', '0.544', '1.447', '0.467'],
        ['ChemBERTa', 'QM7', '6.962', '5.307', '4.593', '4.594'],
        ['DNN', 'ESOL', '1.003', '0.675', '0.107', '0.011'],
        ['DNN', 'FreeSolv', '1.337', '1.754', '0.726', '0.814'],
        ['DNN', 'Lipo', '0.615', '0.616', '0.616', '0.147'],
        ['DNN', 'QM7', '<b>10.298</b>', '<b>5.477</b>', '5.523', '4.554'],
    ], num_cols=(2, 3, 4, 5), cap='<b>Table 5.</b> Deterministic regression NLL, published and reproduced (journal cell c0138).'))
    b.append(table(['Backbone / split', 'z-score SD of errors', 'coverage of 90 % interval', 'mean predicted variance / MSE'], [
        ['ChemBERTa random', '1.086', '0.880', '0.980'], ['ChemBERTa scaffold', '1.373', '0.795', '0.711'],
        ['DNN random', '0.955', '0.922', '1.446'], ['DNN scaffold', '1.286', '0.811', '0.889'],
    ], num_cols=(1, 2, 3), cap='<b>Table 6.</b> Calibration of the learned variances, all methods pooled (journal cell c0138). A z-score SD of 1 and coverage of 0.90 mean calibrated.'))
    b.append("""<p>A consistency check on the published numbers: the published DNN scaffold QM7 row has RMSE 142.9 and NLL 10.30. A Gaussian with constant standard deviation reaching that NLL needs &sigma; &asymp; 39, 3.6 times smaller than the RMSE; a calibrated one would give about 5.5, which is what we obtain (5.48). This is an inference from two published numbers, not a test on published predictions. The authors' repository releases only aggregated report tables (means and SDs per method), no predictions, checkpoints or logs, so the published predictions cannot be examined (note n0055).</p>""")

    b.append('<h2>4. A check against the authors\' own code (DNN)</h2>')
    b.append("""<p>To tell whether the QM7 gap comes from our implementation or from the published run, we cloned the authors' repository at the pinned commit, built its environment from their requirements (<code>torch_scatter</code> and <code>torch_cluster</code> from prebuilt wheels at the same versions; one undeclared dependency, <code>rich</code>, added), and ran their unmodified trainer for the DNN with the settings stated in the paper's Appendix B (GELU, batch 256, learning rate 2e-4, 400 epochs, tolerance 50, 158,082 parameters), seeds 0 to 2, with their own metric functions. The comparison and its decision rule were registered before the runs.</p>""")
    b.append(table(['Dataset / metric', 'Published (SD)', 'Our Tier B', 'Authors\' code, example config', 'Authors\' code, paper settings'], [
        ['ESOL RMSE', '0.905 (0.029)', '0.913', '0.923', '0.906'], ['ESOL NLL', '1.003 (0.377)', '0.675', '0.412', '0.761'], ['ESOL CE', '0.0465 (0.0052)', '0.0465', '0.0363', '0.0475'],
        ['QM7 RMSE', '142.9 (2.9)', '127.3', '137.6', '<b>126.9</b>'], ['QM7 MAE', '95.7 (3.3)', '85.2', '94.1', '84.2'],
        ['QM7 NLL', '10.30 (1.08)', '5.48', '5.78', '<b>5.47</b>'], ['QM7 CE', '0.0393 (0.0036)', '0.0152', '0.0232', '0.0167'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 7.</b> DNN, scaffold, Deterministic (journal cells c0165, c0169, c0181; notes n0054, n0057). The example-config run is superseded by the paper-settings run.'))
    b.append("""<ul><li>With the paper's settings the authors' code reproduces <b>our</b> QM7 numbers (NLL 5.47 against 5.48; RMSE 126.9 against 127.3), within one of our seed SDs on all four metrics, and not the published ones. By the rule registered beforehand, the difference between our Tier B and the published QM7 numbers lies in the published run, not in our reimplementation. ESOL is consistent with the published row under both implementations.</li>
<li><b>Architecture.</b> The paper's Table 5 lists 0.158 M parameters for the DNN. That equals nine linear 128-wide layers (the input layer plus eight hidden layers); our network had eight linear layers in total (141,570 parameters). Depth made no measurable difference on QM7 or ESOL (the authors' 9-layer network matches ours), so the idea that depth explained the QM7 RMSE gap is refuted. The one remaining paper/code difference is the activation (paper: GELU; code default: ReLU).</li>
<li><b>An observation, not a conclusion.</b> The published QM7 scaffold NLL is 5.25 for GIN and 4.62 for Uni-Mol but 10.30 for the DNN; our DNN (5.48) sits with the other backbones, and the published DNN value stands out.</li></ul>""")

    b.append('<h2>5. GIN (2D graph)</h2>')
    b.append("""<p>GIN runs through our pipeline with the authors' GIN encoder (atom embedding, torch_geometric GIN with <code>jk="cat"</code>, global add pooling) and graph featuriser imported, under a protocol frozen before any training: five layers, 128 hidden units, dropout 0.1, batch 128, constant learning rate 1e-4, 200 epochs with tolerance 50, AdamW (weight decay 0.01). The GPU smoke test gave exactly 260,226 parameters, matching the paper and the authors' code. The full grid (336 fits) ran in 2.7 hours with no failure, 32 of 32 groups frozen and evaluated, and an unchanged protocol hash; the pilot had projected at most 3.0 GPU-hours.</p>""")
    b.append(table(['Dataset', 'Metric', 'Published (SD)', 'Ours', 'Difference', 'z'], [
        ['BACE', 'ROC-AUC', '0.731 (0.015)', '0.756', '+3.4 %', '+1.65'], ['BBBP', 'ROC-AUC', '0.628 (0.002)', '0.614', '&minus;2.3 %', '&minus;5.73'],
        ['ClinTox', 'ROC-AUC', '0.696 (0.018)', '0.684', '&minus;1.8 %', '&minus;0.69'], ['Tox21', 'ROC-AUC', '0.671 (0.003)', '0.696', '+3.7 %', '+7.76'],
        ['ESOL', 'RMSE', '1.551 (0.064)', '1.420', '&minus;8.5 %', '&minus;2.06'], ['FreeSolv', 'RMSE', '2.201 (0.275)', '2.275', '+3.4 %', '+0.27'],
        ['Lipo', 'RMSE', '0.929 (0.111)', '0.820', '&minus;11.7 %', '&minus;0.98'], ['QM7', 'RMSE', '122.43 (0.069)', '120.93', '&minus;1.2 %', '<b>&minus;21.74</b>'],
    ], num_cols=(2, 3, 4, 5), cap='<b>Table 8.</b> GIN, scaffold, Deterministic, primary metric in natural units (journal cell c0278). Median absolute difference: 2.8 % (ROC-AUC), 6.0 % (RMSE). Large z-scores come from tiny published SDs; for QM7 the NLL agrees (5.241 against 5.249).'))
    svg = grouped_bars('GIN Deterministic: ours minus published, % of published value', ['BACE', 'BBBP', 'ClinTox', 'Tox21', 'ESOL', 'FreeSolv', 'Lipo', 'QM7'],
                       [('primary metric (ROC-AUC or RMSE)', [3.4, -2.3, -1.8, 3.7, -8.5, 3.4, -11.7, -1.2])], [C['b']], -14, 6, '% difference', h=230, fmt='{:+.1f}', label_vals=True, ticks=[-12, -8, -4, 0, 4])
    b.append(f'<figure>{svg}<div class="cap"><b>Figure 2.</b> Differences in natural units are small to moderate; classification AUC is higher than published on BACE and Tox21, lower on BBBP and ClinTox; regression RMSE is lower than published on ESOL and Lipo.</div></figure>')
    b.append(table(['Method', 'Rows', 'Within 1 SD', 'Within 2 SD', 'Median |z|'], [
        ['SGLD', 38, 16, 29, '1.03'], ['Deterministic', 38, 18, 27, '1.04'], ['SWAG', 38, 14, 29, '1.22'], ['Focal Loss', 24, 6, 19, '1.28'],
        ['MC dropout', 38, 12, 26, '1.33'], ['BBP', 38, 12, 23, '1.51'], ['Temperature', 22, 8, 13, '1.57'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 9.</b> GIN agreement by UQ method (journal cell c0278): no method stands out.'))
    b.append("""<p>QM7 is the worst-agreeing GIN dataset in the counts (3 of 20 within 1 SD); its published RMSE SD is 0.069 (0.06 % of the value) while our own seeds vary by 1.9 (1.6 %), so this is a small-SD artefact, not a failure: the QM7 NLL and CE agree. One genuine anomaly: <b>GIN SGLD diverges on regression for seed 100</b> in our runs (random-1 Lipo RMSE 19,012; FreeSolv 97), and the published GIN tables show comparable instability for SGLD (ESOL RMSE 3.32 &plusmn; 2.12), so this is probably not specific to our pipeline. It was not investigated.</p>""")

    b.append('<h2>5b. Uni-Mol (3D conformers)</h2>')
    b.append("""<p>Uni-Mol runs through our pipeline with the authors' model, dataset and collator classes imported unchanged (the pre-trained checkpoint, eleven conformers per molecule: one sampled per training step, all eleven averaged at inference after the output transform). Our head, training loop, scoring and evaluation are used, under a protocol frozen before any training (hash 8ab19980&hellip;): lr 5e-5, batch 128, 100 epochs, tolerance 40, linear schedule with warm-up, gradient clipping at 1 (the authors' defaults). Scope: scaffold split only, ten Deterministic fits per dataset (seeds 0 to 9; seeds 0 to 2 are the Deterministic runs, all ten the pooled ensemble), with MC dropout and temperature scaling applied afterwards and, under a second frozen addendum (hash fbefa527&hellip;), SWAG, BBP, SGLD and focal-loss fits (seeds 0 to 2: nine further fits per regression and twelve per classification dataset). The 80 Deterministic and ensemble fits cost 14.4 GPU-hours in total (the paper's step time had suggested about 18); the extension fits took about 7 hours on two GPUs. Batches of 128 were realised by gradient accumulation over micro-batches of 32 for ClinTox, Lipo and QM7 and the remaining Tox21 and BBBP fits, after ClinTox ran out of GPU memory with full batches (journal n0113); the gradient is the same.</p>""")
    b.append(table(['Dataset', 'Rows', 'Within 1 SD', 'Within 2 SD'], [
        ['BACE', 42, 27, 38], ['BBBP', 42, 32, 38], ['ClinTox', 42, 20, 29], ['Tox21', 42, 15, 33], ['ESOL', 20, 7, 12], ['FreeSolv', 20, 11, 14], ['Lipo', 20, 6, 10], ['QM7', 20, 11, 14],
        ['<b>Total</b>', '<b>248</b>', '<b>129 (52.0 %)</b>', '<b>188 (75.8 %)</b>'],
    ], num_cols=(1, 2, 3), cap='<b>Table 8a.</b> Uni-Mol agreement with the published scaffold rows, all methods (rows with a published SD; journal cell c0447 job, note n0119 for the first four-method reading). Scaffold only; the scaffold ensemble has no published SD.'))
    b.append(table(['Method', 'Rows', 'Within 1 SD', 'Within 2 SD'], [
        ['MC dropout', 40, '28 (70 %)', '36 (90 %)'], ['Deterministic', 40, '21 (53 %)', '34 (85 %)'], ['Temperature', 24, '12 (50 %)', '23 (96 %)'], ['Focal loss', 24, '14 (58 %)', '19 (79 %)'],
        ['BBP', 40, '24 (60 %)', '29 (73 %)'], ['SWAG', 40, '15 (38 %)', '24 (60 %)'], ['SGLD', 40, '15 (38 %)', '23 (58 %)'],
    ], num_cols=(1, 2, 3), cap='<b>Table 8b.</b> Agreement by method: the simple methods agree best; the sampling methods (SWAG, SGLD) agree least, consistent with their dependence on implementation details.'))
    b.append(table(['Metric', 'Rows', 'Within 1 SD', 'Within 2 SD', 'Median z'], [
        ['ROC-AUC', 28, 11, 25, '&minus;0.07'], ['PRC-AUC', 28, 17, 25, '0.25'], ['ECE', 28, 18, 21, '0.25'], ['Brier', 28, 17, 23, '0.03'], ['NLL (classification)', 28, 14, 21, '0.20'],
        ['RMSE', 20, 9, 13, '&minus;0.52'], ['MAE', 20, 7, 13, '&minus;0.85'], ['NLL (regression)', 20, 10, 13, '&minus;0.77'], ['CE', 20, 9, 11, '&minus;0.11'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 8c.</b> Agreement by metric (all methods): classification agrees well; regression is moderately lower on MAE and NLL.'))
    b.append("""<ul><li><b>Paper claims on Uni-Mol</b> (datasets where the claim holds, published &rarr; reproduced): ensembles best on RMSE or ROC-AUC 5 &rarr; 6 of 8; MC dropout NLL below Deterministic 3 &rarr; 8 of 8; temperature scaling lowers NLL 4 &rarr; 4 of 4; BBP ROC-AUC below Deterministic 1 &rarr; 2 of 4; SGLD ROC-AUC below Deterministic 2 &rarr; 1 of 4. The claims about BBP and SGLD hold on a minority of datasets in both the published and our rows.</li>
<li><b>Bridge check against the authors' unmodified trainer</b> (paper settings; ESOL and BACE seeds 0 to 9, QM7 seeds 0 to 4; registered beforehand): <b>13 of 13</b> comparisons consistent (|d| &lt; 1; largest |d| 0.81). ESOL RMSE 0.851 (authors) against 0.838 (ours), NLL 0.699 against 0.726; BACE ROC-AUC 0.820 against 0.828, ECE 0.245 against 0.257; QM7 NLL 4.397 against 4.479 (journal cell c0414, note n0121). The registered gate (|d| &ge; 2 on more than one dataset) is not triggered. This is a stronger agreement than for GIN and for the first ChemBERTa check.</li>
<li><b>Caveat on the first reading.</b> Over the four methods available first, agreement looked better (58.7 % within 1 SD) because they are the methods that agree best; the full method set gives 52.0 %. The overnight evaluation also had a bug that skipped the new methods (note n0126); the base-method numbers were recomputed and are identical (136 rows compared).</li></ul>""")

    b.append('<h2>6. Bridge checks against the authors\' trainers</h2>')
    b.append('<h3>6.1 GIN: first with three seeds, then with ten</h3>')
    b.append("""<p>The check registered with the GIN protocol compared our GIN Deterministic with the authors' own trainer (paper settings, seeds 0 to 2) on ESOL and QM7: consistent if RMSE, MAE, NLL and CE all differ by less than twice the larger seed SD. Seven of eight cells met the rule; <b>ESOL RMSE did not</b> (authors' trainer 1.562, ours 1.420: 2.07 SD against the limit of 2). That was formally the written reversal condition of the decision to keep our pipeline, so we re-ran it with more seeds.</p>
<p>Ten seeds were available without retraining: our scaffold ensemble members are ten saved Deterministic networks (seeds 0 to 9), evaluated on the test split with the frozen code, against the authors' trainer run for seeds 0 to 9 (Tox21: five). The comparison was registered beforehand: difference of means in units of the pooled seed SD with a Welch 95 % interval; <i>consistent</i> if |d| &lt; 1, <i>different</i> if |d| &ge; 1 and the interval excludes 0.</p>""")
    b.append(table(['Dataset', 'RMSE authors / ours (d)', 'NLL authors / ours (d)'], [
        ['ESOL', '1.536 / 1.602 (&minus;0.40)', '1.123 / 1.434 (&minus;0.85)'], ['FreeSolv', '2.273 / 2.327 (&minus;0.17)', '1.986 / 2.100 (&minus;0.08)'],
        ['Lipo', '0.847 / 0.823 (+0.24)', '0.370 / 0.378 (&minus;0.07)'], ['QM7', '120.6 / 119.8 (+0.37)', '5.281 / 5.248 (+0.28)'],
    ], cap='<b>Table 10.</b> GIN regression bridge, 10 seeds (journal cell c0286): all 16 regression comparisons (RMSE, MAE, NLL, CE on four datasets) are consistent. With ten seeds our ESOL RMSE is 1.602; the three seeds used before (0, 100, 200) happened to give 1.42, and the pooled seed SD is 0.166, more than twice the 0.068 seen with three seeds.'))
    b.append(table(['Dataset', 'Metric', 'Authors', 'Ours', 'd', 'Class'], [
        ['BACE', 'ECE', '0.304', '0.168', '+1.23', 'different'], ['BACE', 'Brier', '0.334', '0.237', '+1.29', 'different'],
        ['ClinTox', 'ROC-AUC', '0.646', '0.692', '&minus;1.04', 'different'], ['ClinTox', 'PRC-AUC', '0.555', '0.601', '&minus;1.24', 'different'],
        ['Tox21 (5 seeds)', 'ROC-AUC', '0.679', '0.692', '&minus;2.73', 'different'], ['Tox21 (5 seeds)', 'NLL', '0.351', '0.357', '&minus;1.33', 'unclear'],
    ], num_cols=(2, 3, 4), cap='<b>Table 11.</b> The classification comparisons that are not consistent (journal cell c0287); the other 14 of 20 are consistent. In all of them our GIN discriminates or calibrates better than the authors\' trainer (by 0.013 to 0.046 AUC, and 0.14 in ECE on BACE). The cause was not investigated; candidates are the head initialisation, the early-stopping comparison and the classification loss path.'))
    b.append("""<p><b>Verdict.</b> 30 of 36 comparisons consistent, 5 different, 1 unclear. The reversal condition of the pipeline decision (more than 2 seed SDs on Deterministic NLL or RMSE) is <b>no longer met</b>, so the decision to continue with our pipeline stands. The classification differences lie outside that condition and are reported as a limitation.</p>""")
    b.append('<h3>6.2 ChemBERTa: the first cross-check</h3>')
    b.append(table(['Dataset', 'Metric', 'Authors\' trainer', 'Ours', 'Diff / larger seed SD', 'Consistent (&lt; 2)'], [
        ['ESOL', 'RMSE', '1.150', '1.090', '4.3', '<span class="pill bad">no</span>'], ['ESOL', 'MAE', '0.865', '0.841', '1.1', '<span class="pill ok">yes</span>'],
        ['ESOL', 'NLL', '0.764', '1.033', '5.0', '<span class="pill bad">no</span>'], ['ESOL', 'CE', '0.0301', '0.0383', '8.8', '<span class="pill bad">no</span>'],
        ['QM7', 'RMSE', '116.9', '115.4', '1.9', '<span class="pill ok">yes</span>'], ['QM7', 'MAE', '72.7', '72.2', '0.4', '<span class="pill ok">yes</span>'],
        ['QM7', 'NLL', '5.105', '5.307', '2.1', '<span class="pill bad">no</span>'], ['QM7', 'CE', '0.0144', '0.0166', '1.0', '<span class="pill ok">yes</span>'],
    ], num_cols=(2, 3, 4), cap='<b>Table 12.</b> Authors\' ChemBERTa trainer (Linear descriptor, pinned ChemBERTa-77M; lr 1e-5, batch 128, 200 epochs, tolerance 40) against our ChemBERTa Deterministic, scaffold, three seeds each (journal cell c0286). <b>Correction:</b> the authors\' trainer was run with a linear schedule and 10 % warm-up, which was our choice (the paper states no schedule for ChemBERTa and the authors\' code default is a constant learning rate), while our pipeline uses a constant rate; so this comparison confounds pipeline and schedule (note n0111).'))
    b.append("""<p>Four of eight cells are consistent. Our ChemBERTa pipeline is therefore <b>not interchangeable</b> with the configuration we ran the authors' trainer in, on calibration metrics (ESOL NLL and CE, QM7 NLL), with only three seeds each. Because that run used a schedule our pipeline does not (see the Table 12 correction), the cause is not established; section 6.5 reports the follow-up test, and the effect of the same schedule on the Laplace gain is in Report 2 (Table 2d). The context matters for the reproduction: the published ChemBERTa ESOL scaffold NLL is 2.28, the authors' own trainer gives 0.764 and ours 1.033, so <b>the published value is not reproduced by their code either</b>, and their trainer is further from the published ESOL RMSE than ours.</p>""")
    b.append('<h3>6.3 Descriptor-version drift (DNN)</h3>')
    b.append(table(['Dataset', 'Molecules', 'Molecules with any feature differing', 'Largest absolute difference'], [
        ['BACE', '1,513', '0', '3e-8'], ['BBBP', '2,039', '0', '3e-8'], ['ClinTox', '1,478', '5 (0.34 %), one feature', '1.0 (one feature)'], ['ESOL', '1,128', '0', '3e-8'],
        ['FreeSolv', '642', '0', '3e-8'], ['Lipo', '4,200', '0', '3e-8'], ['QM7', '6,830', '0', '3e-8'], ['Tox21', '7,831', '16 (0.20 %), 8 of them zero in Tier B', '1.0'],
    ], num_cols=(1,), cap='<b>Table 13.</b> The 200 normalised descriptors recomputed in the authors\' environment (RDKit 2023.09.5, descriptastorus 2.6.1) against the cached Tier-B ones (RDKit 2024.09.6, 2.8.0); registered threshold for material drift: more than 5 % of molecules (journal note n0100).'))
    b.append("""<p>The drift is <b>immaterial</b>: six of eight datasets are identical to float precision. The eight Tox21 SMILES (seven in training, one in test; 0.1 % of the rows) that do not parse in the newer RDKit get zero descriptors in the DNN. No DNN rerun is needed. ChemBERTa uses raw SMILES and is unaffected.</p>""")

    b.append('<h3>6.4 Random splits: identical to the authors\' routine</h3>')
    b.append("""<p>The 3.76 GB Uni-Mol archive (the source of the authors' random splits) was downloaded with the student's approval, and the eight datasets were extracted. Rows of the Uni-Mol LMDB files were concatenated in the order train, valid, test, as in the authors' <code>assist/dataset_build_from_unimol.py</code>, and compared with our scaffold-file concatenation; the authors' two successive <code>train_test_split</code> calls (random_state 0, 1, 2) were then applied to both (journal cell c0306, note n0104).</p>""")
    b.append(table(['Dataset', 'Rows', 'Partition sizes (train / valid / test)', 'Same molecule at every position', 'Same validation and test sets, seeds 0 to 2'], [
        [d, n, s, 'yes', 'yes'] for d, n, s in [('BACE', '1,513', '1,210 / 151 / 152'), ('BBBP', '2,039', '1,631 / 204 / 204'), ('ClinTox', '1,478', '1,182 / 148 / 148'), ('Tox21', '7,831', '6,264 / 783 / 784'),
                                               ('ESOL', '1,128', '902 / 113 / 113'), ('FreeSolv', '642', '513 / 64 / 65'), ('Lipo', '4,200', '3,360 / 420 / 420'), ('QM7', '6,830', '5,464 / 685 / 681')]
    ], num_cols=(1,), cap='<b>Table 14a.</b> Check D. Jaccard overlap of validation and test molecule sets is 1.0 in every case. One assumption remains: the authors\' random-split seeds are 0, 1, 2; their own split files are not released, so a different seed list in their experiments would not be detected.'))
    b.append('<h3>6.5 ChemBERTa: the calibration gap was the learning-rate schedule, not the pipeline</h3>')
    b.append("""<p>After the correction to Table 12 (section 6.2), the ChemBERTa comparison was repeated with matching schedules: the authors' trainer with its own default (constant), and our pipeline with the linear schedule and 10 % warm-up (three seeds per arm, test metrics, journal notes n0111 and n0116). Two further arms checked the head initialisation (PyTorch default instead of our Xavier with bias 0.01).</p>""")
    b.append(table(['Dataset', 'Arm', 'RMSE', 'NLL', 'CE'], [
        ['ESOL', 'authors&rsquo; trainer, linear schedule (earlier bridge)', '1.150 &plusmn; 0.012', '<b>0.764</b> &plusmn; 0.054', '0.0301'],
        ['ESOL', 'authors&rsquo; trainer, constant schedule (their default)', '1.084 &plusmn; 0.013', '<b>1.037</b> &plusmn; 0.082', '0.0385'],
        ['ESOL', 'ours (Tier B), constant schedule', '1.090 &plusmn; 0.014', '<b>1.033</b> &plusmn; 0.027', '0.0383'],
        ['ESOL', 'ours, linear schedule', '1.147 &plusmn; 0.046', '<b>0.758</b> &plusmn; 0.051', '0.0308'],
        ['ESOL', 'ours, PyTorch-default head initialisation (constant)', '1.066 &plusmn; 0.001', '0.985 &plusmn; 0.060', '0.0413'],
        ['QM7', 'authors&rsquo; trainer, linear / constant', '116.9 / 116.4', '5.105 / 5.183', '0.0144 / 0.0157'],
        ['QM7', 'ours, constant (Tier B) / linear', '115.4 / 116.8', '5.307 / 5.101', '0.0166 / 0.0144'],
    ], num_cols=(2, 3, 4), cap='<b>Table 12a.</b> ChemBERTa, scaffold, Deterministic, three seeds per arm (mean &plusmn; SD). Under the same schedule the two pipelines agree (constant: 1.037 against 1.033; linear: 0.764 against 0.758); a linear schedule with warm-up lowers ESOL NLL by about 0.27 and QM7 NLL by about 0.2 in both.'))
    b.append("""<ul><li><b>The earlier &ldquo;pipeline difference&rdquo; is withdrawn for ChemBERTa.</b> The authors' trainer and ours produce the same calibration when run with the same schedule. The gap in Table 12 was a configuration difference introduced by us.</li>
<li><b>The schedule moves the uncertainty metrics a lot.</b> The paper states no schedule for ChemBERTa and the code default is constant, so which one produced the published numbers is unknown; neither arm reaches the published ESOL scaffold NLL of 2.28. Our Tier-B ChemBERTa results use a constant schedule, as the code default does.</li>
<li><b>Head initialisation is not the cause</b> of any gap (ESOL NLL 0.985 against 1.033, inside the seed spread).</li></ul>""")
    b.append(table(['Dataset', 'Metric', 'Authors&rsquo; trainer (10 seeds)', 'Ours, Xavier head', 'Ours, default head (10 seeds)'], [
        ['BACE', 'ECE', '0.304', '0.168', '0.189'], ['BACE', 'NLL', '1.899', '&ndash;', '0.720'], ['BACE', 'ROC-AUC', '0.733', '&ndash;', '0.748'],
        ['ClinTox', 'ROC-AUC', '0.646', '0.692', '0.689'], ['ClinTox', 'ECE', '0.181', '&ndash;', '0.119'],
    ], num_cols=(2, 3, 4), cap='<b>Table 12b.</b> GIN classification with the PyTorch-default head initialisation. The differences from the authors&rsquo; trainer (Table 11) persist, so head initialisation is not their cause; their model-selection metric (ROC-AUC, RMSE for ESOL, FreeSolv and Lipo) is the same as ours. The remaining candidate, the learning-rate schedule, is tested in section 6.6 and explains the differences.'))
    b.append('<h2>6.6 GIN classification: the gap was the learning-rate schedule</h2>')
    b.append("""<p>The ten-seed GIN bridge (Table 11) left five classification comparisons &ldquo;different&rdquo;, all in favour of our GIN, with head initialisation and the selection metric ruled out (Table 12b). Reading the authors' code showed one more candidate: their GIN argument class (<code>muben/args/args_2d.py</code>) overrides the base default and sets a <b>linear learning-rate schedule</b> (warm-up ratio 0.1), whereas the paper's Appendix B states a constant rate for GIN and our pipeline uses a constant rate. The same confound produced the ChemBERTa gap (section 6.5). The launch command of the authors' GIN bridge runs could not be located in the project files, so which schedule they ran with is <b>inferred</b>, not known. A test was therefore registered before any fit (journal n0141, protocol sha256 48b2ae3c&hellip;): retrain our GIN Deterministic with the linear schedule (10 % warm-up; nothing else changed), on BACE, BBBP and ClinTox (seeds 0 to 9) and Tox21 (seeds 0 to 4, as the authors' runs), and repeat the bridge comparison (d = difference over pooled SD, Welch 95 % interval; consistent if |d| &lt; 1). Hypothesis written in advance: at least 18 of 20 comparisons consistent; at most 15 would reject the schedule.</p>""")
    b.append(table(['Dataset', 'Metric', 'Authors&rsquo; trainer', 'Ours, constant schedule', 'Ours, linear schedule'], [
        ['BACE', 'ROC-AUC', '0.7333', '0.7215 (d 0.25; consistent)', '0.735 (d &minus;0.04; consistent)'],
        ['BACE', 'PRC-AUC', '0.7231', '0.7083 (d 0.39; consistent)', '0.7036 (d 0.56; consistent)'],
        ['BACE', 'ECE', '0.3036', '0.1678 (d 1.23; <b>different</b>)', '0.227 (d 0.53; consistent)'],
        ['BACE', 'NLL', '1.8987', '0.7055 (d 0.89; consistent)', '1.1833 (d 0.47; consistent)'],
        ['BACE', 'Brier', '0.3339', '0.2371 (d 1.29; <b>different</b>)', '0.2855 (d 0.48; consistent)'],
        ['BBBP', 'ROC-AUC', '0.6182', '0.6214 (d &minus;0.14; consistent)', '0.6252 (d &minus;0.43; consistent)'],
        ['BBBP', 'PRC-AUC', '0.6048', '0.6148 (d &minus;0.39; consistent)', '0.617 (d &minus;0.63; consistent)'],
        ['BBBP', 'ECE', '0.2459', '0.2407 (d 0.24; consistent)', '0.2416 (d 0.22; consistent)'],
        ['BBBP', 'NLL', '1.0733', '1.0957 (d &minus;0.38; consistent)', '1.0687 (d 0.15; consistent)'],
        ['BBBP', 'Brier', '0.2956', '0.2951 (d 0.05; consistent)', '0.2951 (d 0.07; consistent)'],
        ['ClinTox', 'ROC-AUC', '0.6457', '0.6918 (d &minus;1.04; <b>different</b>)', '0.6359 (d 0.23; consistent)'],
        ['ClinTox', 'PRC-AUC', '0.5553', '0.6014 (d &minus;1.24; <b>different</b>)', '0.5512 (d 0.11; consistent)'],
        ['ClinTox', 'ECE', '0.1813', '0.1213 (d 0.94; consistent)', '0.1879 (d &minus;0.08; consistent)'],
        ['ClinTox', 'NLL', '0.4206', '0.3966 (d 0.32; consistent)', '0.4383 (d &minus;0.21; consistent)'],
        ['ClinTox', 'Brier', '0.1174', '0.0941 (d 0.84; consistent)', '0.1217 (d &minus;0.11; consistent)'],
        ['Tox21', 'ROC-AUC', '0.6789', '0.6919 (d &minus;2.73; <b>different</b>)', '0.6767 (d 0.57; consistent)'],
        ['Tox21', 'PRC-AUC', '0.3171', '0.3265 (d &minus;0.82; consistent)', '0.3096 (d 0.72; consistent)'],
        ['Tox21', 'ECE', '0.0718', '0.0715 (d 0.05; consistent)', '0.072 (d &minus;0.05; consistent)'],
        ['Tox21', 'NLL', '0.3508', '0.3568 (d &minus;1.33; <b>unclear</b>)', '0.3504 (d 0.09; consistent)'],
        ['Tox21', 'Brier', '0.0884', '0.0876 (d 0.3; consistent)', '0.0887 (d &minus;0.1; consistent)']
    ], num_cols=(2,), cap='<b>Table 12c.</b> GIN classification bridge, scaffold, Deterministic (journal note n0144; ten seeds, Tox21 five). With the constant schedule 14 of 20 comparisons are consistent, 5 different and 1 unclear; with the linear schedule <b>all 20 are consistent</b>. The five &ldquo;different&rdquo; rows (BACE ECE and Brier, ClinTox ROC-AUC and PRC-AUC, Tox21 ROC-AUC) all move to |d| &lt; 0.6.'))
    b.append("""<ul><li><b>The hypothesis is supported</b> (20 of 20 against the registered bar of 18): the GIN classification differences were a schedule difference, not a pipeline difference. The earlier statement that their cause was not found is withdrawn.</li>
<li><b>What this implies.</b> The authors' GIN code and the paper's stated settings disagree on the schedule (code default linear, paper constant). Our agreement with their runs suggests the bridge runs used the code default; the published GIN tables may have been produced with either, and a schedule has been shown (here and in Report 2, Table 2d) to change both the calibration metrics and the size of a UQ method's benefit.</li>
<li><b>Limits.</b> Classification datasets only; the regression bridge was already consistent under the constant schedule (16 of 16); the authors' launch command is unknown; the GIN reproduction against the <i>published</i> tables (Tables 8 to 10) was not rerun with the linear schedule, so those agreement counts stay as reported.</li></ul>""")
    b.append('<h2>7. The two cross-backbone claims</h2>')
    b.append("""<p>Before any GIN test result was looked at, two claims from the paper's discussion were registered with operational definitions: <b>(a)</b> models with lower expressiveness are better calibrated (ChemBERTa, the largest, has the largest Deterministic calibration error of the three); <b>(b)</b> the choice of UQ method matters more for larger models (the gap between Deterministic and the best published method is larger for ChemBERTa than for both others). Each is counted per dataset; support needs at least 6 of 8.</p>""")
    b.append(table(['Measure', 'Split type', 'Claim (a) supported on', 'Claim (b) supported on'], [
        ['Calibration error (ECE / CE)', 'scaffold', '2 of 8', '0 of 8'], ['Calibration error (ECE / CE)', 'random', '2 of 8', '0 of 8'],
        ['NLL (robustness view)', 'scaffold', '0 of 8', '1 of 8'], ['NLL (robustness view)', 'random', '2 of 8', '4 of 8'],
    ], cap='<b>Table 14.</b> Outcome (journal notes n0070, n0077): neither claim reaches the registered rule of 6 of 8 in any row.'))
    b.append("""<p><b>With Uni-Mol as a fourth backbone</b> (definitions registered before the computation, note n0129; scaffold only; Uni-Mol, 47.6 M parameters, now plays the largest model). Claim (a): Uni-Mol has the largest Deterministic calibration error on <b>5 of 8</b> datasets (the four regression datasets and BBBP), below the rule of 6; claim (b): its gap between Deterministic and the best method is the largest on <b>2 of 8</b>. On NLL the counts are 2 and 4 of 8. The pattern for regression is consistent with the paper's remark that larger models are poorly calibrated for regression (Uni-Mol has the largest calibration error on all four regression datasets), but the registered rules are not met, and Uni-Mol also has more methods to choose the best from than the first three backbones, which favours claim (b). Descriptive Spearman correlations of parameter count with calibration error (n = 4 per dataset) have a median of 0.4.</p>
<p>The pre-trained ChemBERTa is not the worst-calibrated backbone; the GIN usually has the largest Deterministic ECE on the classification datasets. The three backbones differ in more than size (input type, pre-training), so even a supporting pattern would not identify size as the cause.</p>""")

    b.append('<h2>8. Threats to validity</h2>')
    b.append("""<ul>
<li><b>Random splits.</b> Verified identical to the authors' routine applied to the Uni-Mol source files (section 6.4). The authors' own split files are not released, so the assumption that their seeds were 0, 1, 2 cannot be checked. The QM7 files hold 6,830 compounds, not the 7,160 of the paper's Table 4 (a property of the released files).</li>
<li><b>Scope of the authors'-code checks.</b> DNN: Deterministic, scaffold, ESOL and QM7, three seeds. GIN: Deterministic, scaffold, all datasets, ten seeds. ChemBERTa: Deterministic, scaffold, ESOL and QM7, three seeds. No UQ method other than Deterministic has been cross-checked, and random splits not at all.</li>
<li><b>Statistics.</b> Every count rests on three seeds or splits; there is no test and no multiplicity correction; the published SDs also come from three seeds.</li>
<li><b>Unexplained.</b> The cause of the published DNN QM7 NLL, the schedule that produced the published ChemBERTa uncertainty numbers, and the schedule that produced the published GIN and ChemBERTa uncertainty numbers. The GIN classification differences are traced to the learning-rate schedule (section 6.6); the authors' launch command for those runs is not available, so their schedule is inferred.</li></ul>""")

    b.append('<h2>9. What did not work, and corrections along the way</h2>')
    b.append("""<ul>
<li>A first summary of the Tier-B verdict quoted a range of differences from one backbone's rows only (n0047); a sentence about GIN calibration was written before the counts were compared and had to be corrected (n0076).</li>
<li>The first authors'-code DNN runs used the repository's example configuration, which the paper does not state; they are superseded. One launch of those runs failed silently on a path error before it was caught.</li>
<li>An earlier statement that our random splits "are not the authors'" was too strong; check D later confirmed that they are identical.</li>
<li>The GIN compute projection of 3.2 GPU-hours had been computed by hand; the recorded value is 3.0.</li>
<li>The three-seed ESOL bridge miss was an artefact of too few seeds, which the ten-seed check showed.</li></ul>""")

    b.append('<h2>10. Remaining work</h2>')
    b.append("""<div class="box warn"><ul><li>Done since the last version: the authors' random-split row order (section 6.4), and the statistics and shift-aware conformal checks for the extensions (see Report 2).</li>
<li>A confirmatory replication on fresh splits, with claims pre-registered from the present results.</li>
<li>Bridge the UQ methods other than Deterministic, and rerun the GIN comparison with the published tables under the linear schedule. The ChemBERTa and GIN pipeline differences are resolved (sections 6.5 and 6.6): both were the learning-rate schedule.</li>
<li>Uni-Mol: random splits are not planned; the BBP and SGLD claims could be tested with more seeds (three here).</li>
<li>Out of scope by the student's decision: GROVER (its checkpoint link was unreachable), evidential deep learning.</li></ul></div>""")

    b.extend(code_section(1))
    b.append('<h2 class="pb">Appendix: where each result is recorded</h2>')
    b.append(table(['Topic', 'Journal references'], [
        ['Frozen protocols and decisions', 'n0022 (Tier-B choices), n0067 (continue with our pipeline), n0068 (GIN protocol), n0070 (cross-backbone claims), n0095 and n0096 (fix plan)'],
        ['Agreement with published tables', 'c0242, c0130, c0131, c0248, c0249; notes n0046, n0047, n0075, n0076'],
        ['NLL / CE gap', 'c0137, c0138, c0139; notes n0052, n0055'],
        ['Uni-Mol', 'protocol n0107, cost amendment n0109, memory incident n0113, results n0119, bridge n0121; c0403, c0414'],
        ['Authors\' code, DNN', 'c0156, c0164, c0165, c0169, c0177, c0181; notes n0053, n0054, n0056, n0057'],
        ['GIN', 'c0211, c0213, c0219, c0224 (grid job), c0225, c0278; notes n0068, n0075'],
        ['Bridge checks and drift', 'c0281, c0286, c0287; notes n0099 (GIN, 10 seeds), n0100 (ChemBERTa, drift)'],
        ['Random-split identity (check D)', 'c0297 (download), c0306; note n0104'],
        ['ChemBERTa schedule and head-init arms, GIN head init', 'c0280 (original bridge), c0396, c0399; notes n0111 (correction), n0116 (result)'],
        ['GIN schedule test (section 6.6)', 'lead n0137; registration n0141 (c0524); c0526 (job), c0528; finding n0144'],
        ['Cross-backbone claims', 'n0070 (registration), n0077 (outcome)'],
    ], widths=['28%', '72%']))
    return page('Report 1: Reproduction of MUBen with four backbones',
                'DNN, ChemBERTa and GIN: agreement with the published tables, the NLL/CE gap, checks against the authors\' own code, and threats to validity',
                'Student project on MUBen (molecular uncertainty benchmark) &middot; sc-hub project <code>muben-laplace</code> &middot; status 14 October 2026 &middot; companion: Report 2 (extensions)',
                '\n'.join(b))


# ======================================================================================================================
# REPORT 2
# ======================================================================================================================
LAP = {  # paired NLL difference Laplace minus Deterministic (negative = Laplace better)
    'scaffold': {
        'DNN': [-0.0953, -0.0138, 0.0000, -0.0039, -0.1691, -0.3084, -0.0411, -0.0369],
        'ChemBERTa': [-0.0976, -0.1041, 0.0000, -0.0140, -0.4756, -0.1957, -0.1158, -0.0794],
        'GIN': [-0.0086, -0.1879, -0.0432, -0.0378, -0.1894, 0.0247, -0.0316, -0.0092]},
    'random': {
        'DNN': [-0.0231, -0.0136, -0.1342, -0.0009, -0.0364, 0.0094, -0.0153, 0.0007],
        'ChemBERTa': [-0.0574, -0.0181, -0.0017, -0.0011, -0.0783, 0.0278, -0.0968, -0.0038],
        'GIN': [-0.0051, -0.0420, -0.0185, -0.0077, -0.0725, 0.0119, -0.0232, -0.0031]}}
DS = ['BACE', 'BBBP', 'ClinTox', 'Tox21', 'ESOL', 'FreeSolv', 'Lipo', 'QM7']


def report2():
    b = []
    b.append("""<div class="box"><h3>Summary</h3>
<p><b>Questions.</b> (1) Does last-layer Laplace improve the uncertainty of the reproduced MUBen networks, at lower cost than ensembles, and does this hold across datasets, split types and backbones? (2) Do split-conformal intervals and sets built on these networks keep their 90 % coverage under scaffold shift?</p>
<p><b>Laplace: yes for NLL on scaffold splits, not consistently on random splits, for all three backbones.</b> The paired NLL (Laplace minus Deterministic) improves with all three repeats agreeing in sign on 7 of 8 scaffold datasets for the DNN, 7 of 8 for ChemBERTa and 6 of 8 for GIN (the registered rule needs 6), and on 5, 3 and 3 of 8 random-split datasets, where the rule is not met. The mean scaffold gain is &minus;0.084 (DNN), &minus;0.135 (ChemBERTa) and &minus;0.060 (GIN). The gain in regression comes from wider intervals, not better predictions (the mean is unchanged). Laplace fits in 0.05 to 0.46 s per network and does not replace ensembles on NLL. Novelty-corrected variants add little beyond generic variance inflation. The frozen prior-precision grid was hit at an edge in 53 of 120 (DNN), 45 of 120 (ChemBERTa) and 69 of 120 (GIN) task fits; a wider grid, run on the DNN as a sensitivity analysis, leaves the verdict counts unchanged.</p>
<p><b>Uni-Mol (3D).</b> The same Laplace design on the Uni-Mol networks (scaffold only) improves the paired NLL on <b>all eight datasets with all three seeds agreeing</b> (mean &minus;0.248, larger than for the other backbones; Table 2b). Two cautions: the Laplace features average the eleven conformers, whereas the Deterministic comparator averages transformed outputs (a stated deviation), and 14 of 60 task fits chose a grid edge. The conformal results for Uni-Mol are given below.</p>
<p><b>More repeats (10 to 12 networks per dataset).</b> Running Laplace on every available network (the ten ensemble members, plus seeds 100 and 200 for DNN, ChemBERTa and GIN) shows that the <i>direction</i> of the gain is robust and its <i>per-dataset consistency</i> is weaker than three repeats suggested. The mean NLL difference is negative in all 32 backbone-and-dataset cells, and the dataset-level sign-flip test is significant for every backbone (p = 0.0039; 0.0156 after Holm over the four backbones). Under a stricter rule registered beforehand (at least 90 % of repeats negative and the 95 % interval of the mean below zero), 7 of 8 datasets improve for the DNN, 6 for ChemBERTa, 6 for Uni-Mol and only 4 for GIN, so one registered hypothesis (6 of 8 for every backbone) fails for GIN, and the three-repeat verdicts disagree with the new ones on 2 of 8 datasets for GIN and for Uni-Mol (Table 2c).</p>
<p><b>The size of the Laplace gain depends on the training schedule.</b> Retraining ChemBERTa with a linear learning-rate schedule (the one change; journal note n0133) lowers Deterministic NLL on 7 of 8 datasets (mean &minus;0.075) and cuts the mean Laplace gain from &minus;0.135 to &minus;0.055 (ratio 0.41, registered as &ldquo;shrinks&rdquo; below 0.5). Only 5 of 8 datasets improve with all three seeds (rule 6 of 8 not met; 7 of 8 with the constant schedule). Laplace on the constant-schedule network reaches the Deterministic NLL of the linear-schedule network on 5 of 8 datasets, not on Lipo or QM7 (Table 2d).</p>
<p><b>Conformal on Uni-Mol.</b> The pattern of the other backbones repeats, as registered beforehand (hypothesis: classification coverage at least 5 points below target, regression within 0.88 to 0.93, no contrast reaching 6 of 8). On the scaffold split the coverage at the 0.90 target is 0.81 to 0.84 for classification with every method (Deterministic 0.816; BBBP as low as 0.65) and 0.89 to 0.91 for regression (Deterministic 0.886, Laplace 0.905). The method matters little for set size: Laplace sets are 4.8 % larger and Laplace intervals 5.4 % wider than Deterministic, ensembles 4.9 % narrower for regression (Table 4b). None of the registered contrasts (Laplace against Deterministic, Laplace against ensembles, Novelty Laplace against the constant control) reaches 6 of 8 interpretable datasets, classification or regression, as registered; Mondrian calibration raises the coverage of novel molecules from 0.788 to 0.840 (classification, median cost 2 %) and from 0.859 to 0.904 (regression, median cost 12 to 16 %).</p>
<p><b>Conformal: regression intervals hold, classification sets do not under scaffold shift.</b> Mean test coverage of the Deterministic networks at the 0.90 target: regression 0.92 / 0.90 / 0.92 (ChemBERTa / DNN / GIN) on scaffold; classification 0.82 / 0.79 / 0.75 on scaffold against 0.91 / 0.90 / 0.89 on random splits, also on familiar molecules. Mondrian calibration on novel molecules raises their coverage without reaching 0.90. Conformal on a Laplace base is as wide as on Deterministic (within 1.6 % in regression, 2.4 % in classification); only ensembles are narrower for regression on scaffold (4 to 9 %). No registered contrast reaches 6 of 8. The UQ method matters little for conformal sets; the split type matters a lot.</p>
<p><b>Follow-up checks (section 7).</b> Bootstrap intervals with a Holm-corrected sign-flip test confirm the scaffold NLL gain for all three backbones (Holm p = 0.047 each), but the random-split gain only for GIN. Calibrating conformal sets on half of the test scaffold groups restores classification coverage on scaffold to 0.90 for every backbone and method (from 0.75 to 0.82), at 19 to 35 % larger sets; this is an upper bound, not a validation-only procedure.</p>
<p><b>How sure.</b> All Laplace settings were chosen on validation data only and frozen before test features existed, but the test sets had already been opened for the baselines, so everything is <b>exploratory</b>. The conformal calibration set is the validation split that also served early stopping and Laplace tuning, so the guarantee is approximate. Three repeats per dataset; coverage figures are means over 12 dataset-repeat cases and several test sets are small (FreeSolv has 65 test molecules).</p></div>""")

    b.append("""<div class="box"><h3>Setting</h3><p>The study uses three views of the same compounds: computed descriptors (DNN), the SMILES string (ChemBERTa) and the molecular graph (GIN), compared on identical data and splits and not fused into one model (Report 1). Uncertainty matters in practice because a chemist screening new compounds has to decide which predictions to act on. The scaffold split, in which test compounds have core structures absent from the training set, is the &ldquo;unseen drug&rdquo; case, and it is where Laplace helps most and conformal classification sets lose coverage; the random split is the easier case with close analogues in training.</p></div>""")
    b.append('<h2>1. Last-layer Laplace: design</h2>')
    b.append("""<p>Last-layer Laplace places a Gaussian posterior over the output head of a trained network and keeps the backbone fixed. <b>Classification:</b> an independent affine Bernoulli posterior per task with a probit predictive. <b>Regression:</b> a conditional posterior over the mean head with frozen predicted noise, so the predictive mean is unchanged and only the variance grows. The prior precision &alpha; is chosen per task on validation NLL from a grid fixed in the protocol (10<sup>&minus;4</sup> to 10<sup>4</sup>). Hessians are analytic and checked against autograd at every start. The saved Tier-B and GIN networks are reused; nothing is retrained. The same design was applied three times: DNN (stage 1), ChemBERTa (stage 2, features from the frozen backbone on the GPU, checked to reproduce the saved networks' outputs) and GIN (stage 3, sum-pooled representation, 129 parameters per task and output).</p>
<p><b>Registered before any fit:</b> the primary endpoint is the paired NLL difference to the Deterministic network per dataset and split type; a direction claim needs all three repeats to agree in sign; an overall claim needs the same sign on at least 6 of 8 datasets. All settings were chosen on validation data only, and FROZEN markers (primary and novelty) were written before any test prediction existed. A secondary arm adds a novelty term (variance proportional to the squared distance to the training set) with a constant-inflation control.</p>""")

    b.append('<h2>2. Results: Laplace against Deterministic</h2>')
    sc = LAP['scaffold']
    rd = LAP['random']
    svg1 = grouped_bars('Scaffold splits: NLL, Laplace minus Deterministic (negative = Laplace better)', DS,
                        [('DNN', sc['DNN']), ('ChemBERTa', sc['ChemBERTa']), ('GIN', sc['GIN'])], [C['DNN'], C['ChemBERTa'], C['GIN']], -0.5, 0.1, 'NLL difference', h=250, ticks=[-0.4, -0.3, -0.2, -0.1, 0.0])
    svg2 = grouped_bars('Random splits: NLL, Laplace minus Deterministic', DS,
                        [('DNN', rd['DNN']), ('ChemBERTa', rd['ChemBERTa']), ('GIN', rd['GIN'])], [C['DNN'], C['ChemBERTa'], C['GIN']], -0.15, 0.05, 'NLL difference', h=250, ticks=[-0.15, -0.10, -0.05, 0.0, 0.05])
    b.append(f'<figure>{svg1}{svg2}<div class="cap"><b>Figure 1.</b> Paired NLL difference by dataset (journal cells c0189, c0202, c0251). Scaffold gains are consistent and largest for ChemBERTa (ESOL &minus;0.476); on random splits the effects are small and FreeSolv is slightly worse for all three backbones.</div></figure>')
    rows = []
    for i, d in enumerate(DS):
        rows.append([d] + [f'{sc[bb][i]:+.3f}' for bb in ('DNN', 'ChemBERTa', 'GIN')] + [f'{rd[bb][i]:+.3f}' for bb in ('DNN', 'ChemBERTa', 'GIN')])
    b.append(table(['Dataset', 'DNN scaffold', 'ChemBERTa scaffold', 'GIN scaffold', 'DNN random', 'ChemBERTa random', 'GIN random'], rows, num_cols=(1, 2, 3, 4, 5, 6),
                   cap='<b>Table 1.</b> Mean paired NLL difference (Laplace minus Deterministic). ClinTox on scaffold is exactly 0 for DNN and ChemBERTa because the selected prior precision is at the upper grid edge, which recovers the plain network.'))
    b.append(table(['Backbone', 'Scaffold: improves / mixed / worsens', 'Rule (6 of 8)', 'Random: improves / mixed / worsens', 'Rule (6 of 8)'], [
        ['DNN', '7 / 1 / 0', '<span class="pill ok">met</span>', '5 / 3 / 0', '<span class="pill bad">not met</span>'],
        ['ChemBERTa', '7 / 1 / 0', '<span class="pill ok">met</span>', '3 / 4 / 1', '<span class="pill bad">not met</span>'],
        ['GIN', '6 / 2 / 0', '<span class="pill ok">met (at the threshold)</span>', '3 / 4 / 1', '<span class="pill bad">not met</span>'],
    ], cap='<b>Table 2.</b> Verdict counts: a dataset "improves" only if all three repeats agree in sign (journal cells c0189, c0202, c0251, c0252).'))
    b.append(table(['Dataset', 'BACE', 'BBBP', 'ClinTox', 'Tox21', 'ESOL', 'FreeSolv', 'Lipo', 'QM7'], [
        ['Uni-Mol scaffold NLL difference', '&minus;0.284', '&minus;0.585', '&minus;0.005', '&minus;0.016', '&minus;0.416', '&minus;0.448', '&minus;0.209', '&minus;0.022'],
    ], cap='<b>Table 2b.</b> Uni-Mol, scaffold split: mean paired NLL difference (Laplace minus Deterministic), three seeds; all 24 seed-level differences are negative, 8 of 8 datasets improve (journal cells c0406, c0421). Features use the conformer-averaged pooled representation; the Deterministic comparator uses the authors\' averaging of transformed outputs; 14 of 60 task fits hit a prior-grid edge.'))
    b.append(table(['Backbone', 'Networks', 'Improve / unclear / worsen (rule R10)', 'Improve with the original 3 repeats', 'Mean NLL difference, all repeats (3 repeats)', 'Sign-flip p (Holm over 4)'], [
        ['DNN', 12, '7 / 1 / 0', 7, '&minus;0.077 (&minus;0.084)', '0.0039 (0.0156)'], ['ChemBERTa', 12, '6 / 2 / 0', 7, '&minus;0.126 (&minus;0.135)', '0.0039 (0.0156)'],
        ['GIN', 12, '<b>4</b> / 4 / 0', 6, '&minus;0.144 (&minus;0.060)', '0.0039 (0.0156)'], ['Uni-Mol', 10, '6 / 2 / 0', 8, '&minus;0.221 (&minus;0.248)', '0.0039 (0.0156)'],
    ], num_cols=(1, 3, 4), cap='<b>Table 2c.</b> Laplace on every available network (journal note n0132; protocol written before the runs): R10 = at least 90 % of the repeats negative and the 95 % t-interval of the mean below zero; an overall statement needs 6 of 8 datasets. Unclear datasets, GIN: ESOL, FreeSolv, Lipo, QM7 (58 to 75 % of repeats negative); Uni-Mol: ClinTox, QM7; ChemBERTa: BBBP, ClinTox; DNN: ClinTox (all near zero effect). GIN&rsquo;s larger mean gain comes mainly from FreeSolv (&minus;0.58, interval &minus;1.44 to 0.29).'))
    b.append("""<p>The earlier per-seed Laplace NLL on the original seeds is reproduced to a maximum difference of 2.05e-6 over 96 networks, slightly above the tolerance of 1e-6 written in the protocol; the cause was not investigated. Registered hypotheses: H1 (at least 6 of 8 for every backbone) holds for DNN, ChemBERTa and Uni-Mol and fails for GIN; H2 (the new verdict matches the three-repeat verdict on at least 7 of 8 datasets) holds for DNN and ChemBERTa and fails for GIN and Uni-Mol. The conclusion that Laplace helps on scaffold splits stands on average; the claim that it helps <i>consistently on each dataset</i> needs more repeats than three, especially for GIN on small-effect datasets.</p>""")
    b.append('<h3>Does the Laplace gain depend on the learning-rate schedule? (ChemBERTa)</h3>')
    b.append(table(['Dataset', 'Deterministic NLL, constant', 'Deterministic NLL, linear', 'Laplace gain, constant', 'Laplace gain, linear', 'Ratio', 'Verdict, linear (3 seeds)'], [
        ['BACE', '0.771', '0.754', '&minus;0.098', '&minus;0.084', '0.86', 'improves'], ['BBBP', '0.847', '0.747', '&minus;0.104', '&minus;0.033', '0.31', 'improves'],
        ['ClinTox', '0.098', '0.091', '0.000', '0.000', '&ndash;', 'no effect'], ['Tox21', '0.287', '0.278', '&minus;0.014', '&minus;0.007', '0.53', 'improves'],
        ['ESOL', '1.033', '0.781', '&minus;0.476', '&minus;0.157', '0.33', 'improves'], ['FreeSolv', '1.651', '1.822', '&minus;0.196', '&minus;0.096', '0.49', 'mixed'],
        ['Lipo', '0.544', '0.332', '&minus;0.116', '&minus;0.055', '0.48', 'improves'], ['QM7', '5.307', '5.137', '&minus;0.079', '&minus;0.010', '0.13', 'mixed'],
    ], num_cols=(1, 2, 3, 4, 5), cap='<b>Table 2d.</b> ChemBERTa, scaffold, three seeds (0, 100, 200), constant against linear schedule with 10 % warm-up; the Laplace stage is unchanged and the protocol was written before the runs (note n0131). Mean gain constant &minus;0.135, linear &minus;0.055. Rule: all three seeds negative on at least 6 of 8 datasets: 5 of 8 (constant 7 of 8). Bootstrap intervals over test molecules exclude zero for BACE, BBBP, Tox21, ESOL and Lipo; FreeSolv and QM7 are unclear; the sign-flip test across datasets gives p = 0.008 (exploratory). ClinTox is zero because the selected prior precision is at the upper grid edge.'))
    b.append("""<p>The registered hypothesis (the gain is smaller under the linear schedule on the regression datasets but still improves on at least 6 of 8 datasets) holds in its first part and fails in its second. A schedule that makes the Deterministic network better calibrated removes more than half of the Laplace gain and makes it less consistent: the benefit of a UQ method depends on training details that benchmarks rarely report. Limits: one backbone, three seeds, conformal prediction not rerun under the linear schedule.</p>""")
    b.append("""<ul>
<li><b>Uni-Mol statistics and control.</b> Bootstrap intervals over test molecules (2,000 resamples) lie below zero for all eight datasets (closest: ClinTox, &minus;0.0046 with interval [&minus;0.0099, 0.0000]); the exact one-sided sign-flip test over the eight datasets gives p = 0.0039, the smallest value attainable, and p = 0.027 after Holm correction over the seven backbone-and-split tests (journal note n0124). A posterior-free control (the same conformer-averaged features, zero covariance) separates the stated averaging deviation from the posterior: averaging alone changes test NLL by +0.094 (BACE), +0.112 (BBBP) and +0.041 (Lipo), that is slightly <i>worse</i>, and by at most 0.003 on the other five datasets, while the posterior changes it by &minus;0.005 to &minus;0.697 on all eight. The deviation did not create the gain; on BACE, BBBP and Lipo it hides part of it (note n0130).</li>
<li><b>Other metrics</b> (mean over datasets, scaffold / random). DNN: ROC-AUC &minus;0.010 / &minus;0.002, ECE &minus;0.006 / &minus;0.004, Brier &minus;0.0022 / &minus;0.0002, regression CE &minus;0.002 / &minus;0.0008. ChemBERTa: ROC-AUC &minus;0.010 / &minus;0.0006, ECE &minus;0.029 / &minus;0.002, Brier &minus;0.010 / &minus;0.002, CE &minus;0.0105 / &minus;0.003. RMSE and MAE are unchanged by construction. Calibration gains on scaffold are largest for the pre-trained model (ECE about five times the DNN's).</li>
<li><b>Against other methods.</b> Against MC dropout the DNN improves on 5 of 8 scaffold datasets (rule not met), ChemBERTa on 3 with 3 worse; on ChemBERTa random splits MC dropout has the lower NLL on Lipo (by 0.18) and QM7 (by 0.13). Against ensembles Laplace is worse on 4 of 8 scaffold datasets for the DNN (QM7 +0.264, FreeSolv +0.176), better on 4 and worse on 3 for ChemBERTa, and worse on 3 with 5 mixed for GIN: <b>Laplace improves</b> on the single network cheaply but does not replace ensembles on NLL.</b></li>
<li><b>Cost.</b> Laplace fit: median 0.05 s (DNN), 0.13 s (GIN), 0.46 s (ChemBERTa) per network, against 10 networks for the scaffold ensemble.</li>
<li><b>Why regression improves.</b> The Deterministic variances are overconfident on scaffold splits (z-score SD about 1.3, Report 1); Laplace adds posterior variance and leaves the mean untouched, so NLL improves through wider intervals, not better predictions.</li></ul>""")

    b.append('<h2>3. The prior-precision grid and a sensitivity analysis</h2>')
    b.append("""<p>The selected &alpha; sits on an edge of the frozen grid in 53 of 120 DNN task fits (37 at 10<sup>&minus;4</sup>, 16 at 10<sup>4</sup>; regression 20 of 24), 45 of 120 ChemBERTa fits (27 and 18) and 69 of 120 GIN fits (59 and 10). An upper-edge pick means "no posterior effect" (the plain network); a lower-edge pick means the optimum may lie below the grid. The student decided not to widen the primary grid, so a wider grid (10<sup>&minus;8</sup> to 10<sup>4</sup>) was run on the DNN as a registered, labelled sensitivity analysis with the same validation-only rules.</p>""")
    b.append(table(['Dataset', 'Scaffold, frozen grid', 'Scaffold, wide grid', 'Random, frozen grid', 'Random, wide grid'], [
        ['BACE', '&minus;0.095', '&minus;0.150', '&minus;0.023', '&minus;0.023'], ['BBBP', '&minus;0.014', '&minus;0.014', '&minus;0.014', '&minus;0.014'],
        ['ClinTox', '0.000', '0.000', '&minus;0.134', '&minus;0.134'], ['Tox21', '&minus;0.004', '&minus;0.004', '&minus;0.001', '&minus;0.001'],
        ['ESOL', '&minus;0.169', '&minus;0.208', '&minus;0.036', '&minus;0.043'], ['FreeSolv', '&minus;0.308', '&minus;0.393', '+0.009', '+0.016'],
        ['Lipo', '&minus;0.041', '&minus;0.042', '&minus;0.015', '&minus;0.016'], ['QM7', '&minus;0.037', '&minus;0.037', '+0.001', '+0.001'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 3.</b> DNN: NLL difference to Deterministic with the frozen and the wide grid (journal cell c0192). Verdict counts are identical; three scaffold gains grow by 0.04 to 0.085.'))
    b.append("""<p>By the registered edge criterion the problem is <b>not resolved</b> (32 of 120 fits, 26.7 %, still choose the new lower edge, against a 10 % threshold), but the validation grids show a plateau: for those fits the NLL at &alpha; = 10<sup>&minus;6</sup> and 10<sup>&minus;8</sup> differs by about 10<sup>&minus;5</sup> (regression) and 5&times;10<sup>&minus;7</sup> (classification), and 84 % differ by less than 10<sup>&minus;3</sup>. Below about 10<sup>&minus;6</sup> the prior no longer matters. The test conclusions do not change, and the frozen-grid result stays primary. The sensitivity run was done for the DNN only; for GIN and ChemBERTa a better &alpha; below 10<sup>&minus;4</sup> cannot be excluded.</p>""")

    b.append('<h2>4. Novelty-corrected variants</h2>')
    b.append("""<p>Three variants add to the Laplace variance a term proportional to the squared distance to the training set (maximum Tanimoto similarity, Morgan radius 2, 2048 bits; &lambda; chosen on validation after &alpha;): <i>Laplace constant</i> (a constant inflation, the control), <i>Novelty only</i> and <i>Novelty Laplace</i>. On the DNN, constant inflation alone lowers NLL against plain Laplace on 5 of 8 scaffold datasets (Lipo &minus;0.318, QM7 &minus;0.154, ESOL &minus;0.154, BACE &minus;0.106), and Novelty Laplace beats the constant on only 3 (Lipo &minus;0.068, QM7 &minus;0.046, BBBP &minus;0.023). On ChemBERTa the novelty-specific part adds almost nothing (1 lower, 2 higher, 5 mixed; only Lipo changes: NLL on novel molecules 0.874 to 0.418), and on GIN neither inflation variant gives a consistent gain. <b>Most of the gain is generic variance inflation; the novelty-specific benefit is small and inconsistent</b> (hypothesis-level only).</p>""")

    b.append('<h2>5. Conformal prediction</h2>')
    b.append('<h3>5.1 Design (protocol frozen before any test computation)</h3>')
    b.append("""<p>Split conformal prediction turns a predictive distribution into an interval or set with a promised coverage, here 90 % (secondary: 80 %). Regression score: absolute error divided by the model's predicted standard deviation (an unnormalised score is the baseline). Classification score: one minus the probability of the true class (LAC). The calibration set is the full validation split of each run; coverage and size are measured on the untouched test split. A molecule is <i>novel</i> if its maximum Tanimoto similarity to the training set is below 0.4; Mondrian calibration repeats the procedure inside the novel and familiar groups (at least 20 calibration molecules per group). The amended protocol covers three backbones, every published UQ method and the Laplace family of each, 8 datasets, scaffold and three random splits (6 prediction families &times; 32 groups). <b>Questions:</b> Q1 which method gives the narrowest intervals or smallest sets at equal coverage; Q2 does coverage hold under scaffold shift and on novel molecules; Q3 does Mondrian conditioning restore subgroup coverage and at what cost; Q4 does a Laplace base give narrower intervals or smaller sets than Deterministic and ensembles; Q5 how do coverage and size of the Deterministic network under shift compare across backbones. <b>Primary contrasts:</b> Laplace minus Deterministic, Laplace minus ensembles, and (scaffold) Novelty Laplace minus Laplace constant; a sign counts only if all repeats agree; a width comparison counts only where both methods hold coverage; an overall statement needs 6 of 8 interpretable datasets.</p>""")
    b.append("""<p><b>Checks before the run.</b> The analysis code crashed on the eight Tox21 molecules RDKit cannot parse; it was fixed in a versioned copy that treats them as maximally novel. Reading the code showed that the full run would have silently skipped every GIN group (test files were looked up in two folders that do not hold GIN results); a one-line path fix and an input-coverage check that fails loudly were added before the freeze. ChemBERTa validation predictions had silently run on the CPU through an import order that hides the GPU, taking 20 to 60 times longer; they were recomputed on the GPU. Before the freeze: all six families had 32 of 32 groups; a self-test on validation halves (test untouched) gave mean coverage 0.90 to 0.92; and the code agreed with the <i>crepes</i> library (regression identical; classification identical except where (n+1)(1&minus;&alpha;) is an integer).</p>""")
    b.append('<h3>5.2 Coverage under scaffold shift (Q2, Q5)</h3>')
    b.append(table(['Split', 'Backbone', 'Regression coverage', 'Familiar / novel', 'Classification coverage', 'Familiar / novel'], [
        ['scaffold', 'ChemBERTa', '0.919', '0.936 / 0.900', '<b>0.819</b>', '0.820 / 0.834'], ['scaffold', 'DNN', '0.900', '0.935 / 0.883', '<b>0.788</b>', '0.804 / 0.785'],
        ['scaffold', 'GIN', '0.921', '0.933 / 0.906', '<b>0.753</b>', '0.785 / 0.724'], ['random', 'ChemBERTa', '0.924', '0.933 / 0.875', '0.910', '0.909 / 0.928'],
        ['random', 'DNN', '0.917', '0.925 / 0.879', '0.900', '0.905 / 0.875'], ['random', 'GIN', '0.909', '0.923 / 0.841', '0.886', '0.903 / 0.816'],
    ], num_cols=(2, 4), cap='<b>Table 4.</b> Deterministic networks, mean test coverage at the 0.90 target over 12 dataset-repeat cases (journal cell c0267). Several test sets are small, so single cases are noisy.'))
    b.append('<figure><img src="figs/fig-003.png"><div class="cap"><b>Figure 2.</b> Coverage on novel against familiar test molecules for every backbone (colour) and method class (marker); dashed lines mark the 0.90 target (journal cell c0272). Regression points sit near or above 0.90 on scaffold splits and lower on novel molecules on random splits. Classification scaffold points are far below target on both axes for every method.</div></figure>')
    b.append("""<ul><li><b>Split conformal behaves as theory says under exchangeability</b> (random splits, about 0.90) and, for regression, even under scaffold shift.</li>
<li><b>Classification sets lose 9 to 15 points on scaffold splits</b> (0.75 to 0.82), and familiar molecules are under-covered too, so this is a calibration-set problem, not only a novelty effect: the validation scaffolds do not look like the test scaffolds. The same holds across all UQ methods, so better-calibrated probabilities do not repair a shift between calibration and test molecules.</li>
<li><b>Mondrian calibration</b> raises coverage of novel molecules on scaffold splits for classification from 0.795 to 0.858 (ChemBERTa), 0.751 to 0.829 (DNN) and 0.707 to 0.796 (GIN) without reaching 0.90, and for regression from 0.900 to 0.933, 0.883 to 0.910 and 0.906 to 0.937 at a median width cost of 14 %, 12 % and 7 %. The cost is measured against the overall marginal width, which is not a like-for-like comparison.</li></ul>""")
    b.append(table(['Method', 'Classification coverage', 'Set size / Deterministic', 'Regression coverage', 'Width / Deterministic'], [
        ['Deterministic', '0.816', '1.000', '0.886', '1.000'], ['MC dropout', '0.817', '1.000', '0.891', '0.998'], ['Temperature', '0.820', '1.008', '&ndash;', '&ndash;'], ['Focal loss', '0.843', '1.060', '&ndash;', '&ndash;'],
        ['BBP', '0.810', '1.010', '0.886', '1.008'], ['SWAG', '0.820', '1.011', '0.907', '1.172'], ['SGLD', '0.820', '1.015', '0.898', '1.049'], ['Ensembles', '0.814', '0.971', '0.894', '0.951'],
        ['Laplace', '0.830', '1.048', '0.905', '1.054'], ['Laplace constant', '0.826', '1.032', '0.904', '1.058'], ['Novelty Laplace', '0.826', '1.042', '0.904', '1.066'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 4b.</b> Uni-Mol, scaffold split, target 0.90 (journal note n0126 and the post-processing job; protocol PROTOCOL_unimol.md): mean coverage over datasets and seeds, and size or width as a geometric mean ratio to Deterministic. Classification coverage is below the target for every method; the UQ method changes sizes by 1 to 6 %.'))
    b.append('<h3>5.3 Does a Laplace base help? (Q1, Q4)</h3>')
    b.append('<figure><img src="figs/fig-001.png"><div class="cap"><b>Figure 3.</b> Interval width relative to the Deterministic network (regression, normalised score; geometric mean over datasets and repeats). Laplace variants sit at about 1.00; ensembles are narrower on scaffold (0.91 to 0.96). The clipped bar is GIN SGLD on random splits (3.6), caused by one diverged run.</div></figure>')
    b.append('<figure><img src="figs/fig-002.png"><div class="cap"><b>Figure 4.</b> Prediction-set size relative to the Deterministic network (classification, LAC). Differences between methods are small except focal loss, SWAG and SGLD for GIN on scaffold.</div></figure>')
    b.append("""<p>No row reaches an overall statement under the registered rule. Laplace-based intervals and sets are within 1.6 % (regression, ratios 0.984 to 1.006) and 2.4 % (classification, 0.998 to 1.024) of the Deterministic ones, and the novelty-aware variants do not differ from the constant control in a consistent way. The one clear pattern is that <b>ensembles give narrower regression intervals on scaffold</b> (ratio 0.957 / 0.928 / 0.911 for ChemBERTa / DNN / GIN, from only 4 cases per row because the scaffold ensemble is a single pooled one). Because the interval is the model's own standard deviation times a calibrated quantile, a variance estimate that only rescales the predictions changes little once the scores are normalised, which fits Laplace changing widths so little.</p>""")
    b.append("""<div class="box warn"><h3>A result in its own right: GIN SGLD</h3><p>GIN SGLD on random splits has a mean interval width 3.6 times the Deterministic one. It comes from one run, seed 100 (random-1), where SGLD diverged on regression (Lipo RMSE 19,012 against 0.85 for Deterministic; FreeSolv 97 against 1.7); the same seed degrades scaffold ESOL and Lipo (RMSE 2.8 and 2.2 against 1.4 and 0.9). Seeds 0 and 200 are normal, and the published GIN tables show comparable instability for SGLD on regression (ESOL RMSE 3.32 &plusmn; 2.12), so it is not specific to our pipeline. The conformal contrasts do not involve SGLD. The cause was not investigated.</p></div>""")

    b.append('<h2>6. What did not work, and how sure we are</h2>')
    b.append("""<ul>
<li>The frozen &alpha; grid was too narrow for many fits, which is why the sensitivity analysis exists; it was first run for the DNN only and then for all four backbones (section 8.2).</li>
<li>Novelty-corrected variants were not a clear success, and Laplace does not reach ensembles on NLL on several datasets.</li>
<li>A conformal summary script written before the run applied a decision rule that could never be satisfied (it applied the 6-of-8 rule to four datasets at a time); it was found after the run and handled by an add-on without changing any rule or prediction. The first reading of the Mondrian and coverage tables contained small range errors, corrected in the journal.</li>
<li>All Laplace and conformal results are exploratory: no setting was chosen on test, but the test sets were not fresh. Three repeats for the conformal analysis and the first Laplace stages (10 to 12 networks for the Laplace follow-up in Table 2c); early-stopped (not exact MAP) posterior centres; the conformal calibration set was also used for early stopping and Laplace tuning, so the guarantee is approximate.</li>
<li>Width comparisons are interpretable on only 1 to 5 of 8 datasets per row because coverage often does not hold in every repeat.</li></ul>""")

    b.append('<h2>7. Follow-up checks (registered in n0096, now run)</h2>')
    b.append('<h3>7.1 Check E: statistics for the Laplace results</h3>')
    b.append("""<p>For every dataset the paired NLL difference (Laplace minus Deterministic) was computed per molecule on the saved test predictions (reproducing the stored NLL first), with a 95 % bootstrap interval (2,000 resamples of test molecules, shared across seeds on scaffold). A dataset counts as <i>improves</i> if its interval excludes zero on the improving side. Per backbone and split type, an exact sign-flip test on the 8 dataset z-scores (mean difference over bootstrap SE), one-sided for improvement, with Holm correction over the 6 tests (journal cell c0300, note n0102; wording corrected in n0103).</p>""")
    b.append(table(['Backbone', 'Split', 'Improves / unclear / worsens / identical', 'Sign-flip p', 'Holm p', 'Significant at 0.05'], [
        ['DNN', 'scaffold', '7 / 0 / 0 / 1', '0.0078', '0.047', '<span class="pill ok">yes</span>'],
        ['ChemBERTa', 'scaffold', '6 / 1 / 0 / 1', '0.0078', '0.047', '<span class="pill ok">yes</span>'],
        ['GIN', 'scaffold', '6 / 2 / 0 / 0', '0.0078', '0.047', '<span class="pill ok">yes</span>'],
        ['GIN', 'random', '4 / 4 / 0 / 0', '0.0117', '0.047', '<span class="pill ok">yes</span>'],
        ['DNN', 'random', '4 / 4 / 0 / 0', '0.0312', '0.062', '<span class="pill bad">no</span>'],
        ['ChemBERTa', 'random', '3 / 4 / 1 / 0', '0.0586', '0.062', '<span class="pill bad">no</span>'],
    ], num_cols=(3, 4), cap='<b>Table 5.</b> Check E. "Identical" is ClinTox on scaffold, where the selected prior precision is at the upper grid edge and the predictions equal the plain network (z = 0). With 8 datasets the smallest attainable one-sided p is 0.0039, so Holm correction leaves little room.'))
    b.append("""<ul><li>The <b>scaffold gain is robust</b> to molecule-level uncertainty and multiplicity for all three backbones. On <b>random splits</b> only GIN is supported; the earlier reading that DNN and ChemBERTa show no consistent random-split gain stands.</li>
<li>The per-dataset count under molecule-level intervals (6, 7, 6 of 8 on scaffold) is the same as the sign-agreement counts of Table 2. These are still exploratory: the test sets were opened earlier and there are three repeats.</li></ul>""")
    b.append('<h3>7.2 Check F: shift-aware conformal calibration</h3>')
    b.append("""<p>Test molecules were grouped by Bemis-Murcko scaffold; scaffold groups were assigned at random 50/50 to a calibration half and an evaluation half (50 assignments, both directions, at least 20 calibration molecules). Split conformal (target 0.90) was calibrated on the test calibration half and evaluated on the other half, and compared with the validation-calibrated sets on the same evaluation halves. This is an <b>upper bound</b> on what calibration on labelled molecules from the new domain can do, not a procedure that uses validation data only (journal cell c0298, note n0101).</p>""")
    b.append(table(['Backbone', 'Task / split', 'Coverage, test-half calibration', 'Coverage, validation calibration', 'Size ratio (test-half / validation)'], [
        ['ChemBERTa', 'classification, scaffold', '0.904', '0.819', '1.20'], ['DNN', 'classification, scaffold', '0.904', '0.788', '1.23'], ['GIN', 'classification, scaffold', '0.904', '0.754', '1.35'],
        ['ChemBERTa', 'regression, scaffold', '0.907', '0.919', '1.07'], ['DNN', 'regression, scaffold', '0.906', '0.900', '1.13'], ['GIN', 'regression, scaffold', '0.906', '0.921', '0.86'],
        ['ChemBERTa / DNN / GIN', 'classification, random', '0.901 / 0.903 / 0.902', '0.909 / 0.898 / 0.884', '1.00 / 1.03 / 1.05'],
    ], num_cols=(2, 3, 4), cap='<b>Table 6.</b> Check F, Deterministic networks (means over datasets and the 50 assignments). Laplace and ensembles give the same pattern (classification scaffold coverage 0.904 to 0.905 for every method; ensembles remain the narrowest for regression).'))
    b.append("""<ul><li>The registered hypothesis holds: classification coverage on scaffold goes from <b>0.75 to 0.82 to 0.90</b> on every backbone and method, and regression stays in 0.903 to 0.907 (target range 0.88 to 0.93).</li>
<li>The price is larger sets: <b>19 to 35 %</b> on scaffold for classification; regression intervals widen 6 to 15 % for the DNN and ChemBERTa and shrink 14 to 19 % for GIN. On random splits nothing changes.</li>
<li><b>Uni-Mol (same check, three methods).</b> Classification coverage on scaffold is restored from 0.816 / 0.814 / 0.830 (Deterministic / Ensembles / Laplace) to 0.905 / 0.905 / 0.906 by calibrating on half of the test scaffold groups, at sets 13 to 21 % larger; regression stays at 0.903 to 0.905 with intervals 12 to 14 % wider (note n0130).</li>
<li>The ranking of methods does not change: a Laplace base is within about 1 % of Deterministic, and ensembles are narrower only for regression. The shortfall of classification sets under scaffold shift is therefore a calibration-set problem, not a model-quality problem.</li></ul>""")
    b.append('<h3>7.3 Still open</h3>')
    b.append("""<div class="box warn"><ul><li><b>Fresh-data confirmation</b> (new splits with the Laplace and conformal claims pre-registered from these results) would remove the exploratory label; it needs its own protocol and training.</li>
<li>The wider prior-precision grid for GIN, ChemBERTa and Uni-Mol is done (section 8.2); checks of the GIN SGLD divergence remain open; the conformal contrasts remain unfulfilled at the 6-of-8 rule.</li></ul></div>""")

    b.append('<h2>8. Round 1 follow-ups (14 October)</h2>')
    b.append("""<p>Registered before the runs (journal n0135, protocol sha256 41e645e2&hellip;, and n0138, sha256 150be162&hellip;, with an addendum for the cross-fitted weights); all from saved networks and saved predictions, no new test predictions beyond the saved networks' own; exploratory like everything else here.</p>""")
    b.append('<h3>8.1 Laplace against a one-parameter recalibration</h3>')
    b.append("""<p>For each of the same networks (12 per dataset for DNN, ChemBERTa and GIN, 10 for Uni-Mol; scaffold), a recalibration baseline was fitted on validation data only: in regression one variance multiplier per task (the closed-form maximum-likelihood value, mean and noise head unchanged); in classification one temperature per task. Laplace and the baseline were compared per dataset with rule R10 on the paired NLL difference.</p>""")
    b.append(table(['Backbone', 'Laplace better / scaling better / unclear (of 8)', 'Mean NLL change, Laplace', 'Mean NLL change, scaling', 'Mean Laplace minus scaling', 'Sign-flip p (Laplace better)'], [
        ['DNN', '0 / 5 / 3', '&minus;0.077', '&minus;0.159', '+0.082', '0.95'], ['ChemBERTa', '1 / 3 / 4', '&minus;0.126', '&minus;0.141', '+0.016', '0.82'],
        ['GIN', '2 / 0 / 6', '&minus;0.144', '&minus;0.081', '&minus;0.063', '0.016'], ['Uni-Mol', '0 / 2 / 6', '&minus;0.221', '&minus;0.276', '+0.056', '0.91'],
    ], num_cols=(1, 2, 3, 4, 5), cap='<b>Table 7.</b> Laplace against the one-parameter recalibration (journal note n0143; cells c0527 to c0529). Negative &ldquo;Laplace minus scaling&rdquo; favours Laplace. The registered hypothesis (Laplace is better on at least 6 of 8 datasets for at most two of four backbones) is supported: it is better on none.'))
    b.append("""<ul><li><b>Most of the Laplace gain is reproduced by one fitted scalar</b> on the DNN (scaling wins on BACE, Tox21, ESOL, Lipo and QM7), on ChemBERTa (Tox21, FreeSolv, Lipo) and on Uni-Mol (ESOL, Lipo). It does not reach Laplace on BBBP, where temperature scaling can hurt (DNN +0.048 against Deterministic), and Laplace is clearly better only on ChemBERTa ESOL and on GIN ClinTox and Tox21.</li>
<li><b>GIN is the exception on average</b> (Laplace minus scaling &minus;0.063), mainly because the temperature or variance multiplier hurts on several GIN datasets (for example Lipo +0.168 against Deterministic).</li>
<li><b>Reading.</b> The Laplace NLL gain is largely a generic widening of overconfident variances, which a single parameter also delivers; this fits the novelty-variant result of section 4. Laplace remains cheap and needs no extra tuning, and it matches or is more robust than rescaling on some datasets, but the evidence does not support a claim that its molecule-specific posterior is what helps.</li></ul>""")
    b.append('<h3>8.2 A wider prior grid for all four backbones</h3>')
    b.append("""<p>The same networks were also fitted with the wide grid (10<sup>&minus;8</sup> to 10<sup>4</sup>; Cholesky with an eigen-decomposition fallback), alpha chosen on validation only.</p>""")
    b.append(table(['Backbone', 'Verdict vs Deterministic unchanged (of 8)', 'Mean NLL change from the wide grid', 'Task fits at a grid edge, frozen grid', 'Task fits at a grid edge, wide grid'], [
        ['DNN', '8', '&minus;0.014 (BACE &minus;0.059)', '52 %', '45 %'], ['ChemBERTa', '8', '&minus;0.0003', '42 %', '42 %'],
        ['GIN', '8', '0.0000', '64 %', '64 %'], ['Uni-Mol', '8', '&minus;0.0002', '26 %', '26 %'],
    ], num_cols=(1, 2, 3, 4), cap='<b>Table 8.</b> Wide-grid sensitivity (journal note n0143). Hypothesis H2 (verdict unchanged on at least 7 of 8 datasets for every backbone) is supported. The primary Laplace arm reproduced the item-B per-network NLL to 4.5&times;10<sup>&minus;12</sup> or better for ChemBERTa, GIN and Uni-Mol and to 2.05&times;10<sup>&minus;6</sup> for the DNN (the same marginal exceedance of the 1e-6 tolerance as in item B).'))
    b.append("""<p>The wide grid changes the conclusions nowhere and the test NLL by less than 0.001 except on the DNN (mean &minus;0.014, mostly BACE). Many fits still sit at an edge, mainly the upper one, which means &ldquo;no posterior effect&rdquo; (for example ClinTox). The open question from section 3 is therefore closed for all four backbones: a better prior strength below 10<sup>&minus;4</sup> would not change the picture.</p>""")
    b.append('<h3>8.3 Conformal classification with a few labelled molecules, and with similarity weights</h3>')
    b.append("""<p><b>C1 (few labels).</b> Whole test scaffold groups were moved into a labelled set until it held at least k molecules (k = 5 to 100; 50 random orderings), and sets were evaluated on molecules of the remaining, scaffold-disjoint groups. Three arms: pooled (validation and labelled scores together), balanced (half the weight on the labelled scores) and target-only (labelled scores alone, k &ge; 20). <b>C2 (weighting).</b> Weighted conformal prediction for covariate shift, with weights from a cross-fitted logistic regression on Morgan bits that separates validation from test molecules (features only, no labels), clipped odds; a quantile per test molecule.</p>""")
    b.append(table(['Backbone', 'Validation calibration', 'Pooled, k = 50 / 100', 'Balanced, k = 20 / 50', 'Target-only, k = 20 / 50', 'Weighted (C = 1) / (C = 0.1)'], [
        ['DNN', '0.788', '0.833 / 0.857', '0.861 / 0.874', '0.897 / 0.909', '0.886 / 0.880'], ['ChemBERTa', '0.819', '0.864 / 0.878', '0.882 / 0.889', '0.894 / 0.908', '0.913 / 0.900'],
        ['GIN', '0.753', '0.826 / 0.851', '0.861 / 0.869', '0.901 / 0.910', '0.856 / 0.847'],
    ], num_cols=(1, 2, 3, 4, 5), cap='<b>Table 9.</b> Classification coverage on scaffold (target 0.90; Deterministic, mean over datasets; Laplace is within 0.01; journal note n0139). Size ratio against validation calibration: target-only 1.16 to 1.22 (DNN, ChemBERTa) and about 1.33 (GIN), balanced 1.14 (DNN, ChemBERTa) to 1.23 to 1.27 (GIN), weighted 1.35 to 1.42. Regression coverage stays near 0.90 in every arm; weighted regression intervals are 0.98 (ChemBERTa), 1.25 (DNN) and 1.73 (GIN) times as wide.'))
    b.append("""<ul><li><b>About 20 labelled molecules from the target scaffolds, used alone, restore coverage</b> to 0.89 to 0.90 on all three backbones, for Deterministic and Laplace, at 16 to 33 % larger sets. By the rule registered beforehand this is a deployable correction. Pooling them with the validation scores is slower (0.83 to 0.88 at 50 to 100 molecules) because the validation scores dominate. The registered hypothesis for the balanced arm (0.88 at k &le; 50 on at least two of three backbones) is <b>rejected</b>: only ChemBERTa reaches it.</li>
<li><b>Similarity weighting helps without labels but is expensive.</b> Coverage rises to 0.86 to 0.91, so the registered hypothesis that weighting would not rescue it (staying below 0.85 on at least two backbones) is <b>rejected</b>; but the sets are 35 to 42 % larger, 16 to 24 % of the test molecule-task pairs get the trivial full label set (infinite quantile), and the Kish effective sample size of the validation weights is only 15 to 222. It is a diagnostic of how separable the validation and test molecules are, not a clean fix.</li>
<li>Caveat for k: for multi-task Tox21 fewer than 20 labelled molecules per task can be observed, and those tasks are skipped in the target-only arm.</li></ul>""")
    b.append('<h3>8.4 Conformal under the linear learning-rate schedule</h3>')
    b.append(table(['Task', 'Method', 'Constant schedule: coverage (size)', 'Linear schedule: coverage (size)'], [
        ['classification', 'Deterministic', '0.819 (1.081)', '0.817 (1.080)'], ['classification', 'Laplace', '0.824 (1.114)', '0.826 (1.119)'],
        ['regression', 'Deterministic', '0.919 (101.0)', '0.930 (102.2)'], ['regression', 'Laplace', '0.926 (95.7)', '0.931 (96.1)'],
    ], num_cols=(2, 3), cap='<b>Table 10.</b> ChemBERTa, scaffold, three seeds; validation predictions from the saved linear-schedule networks and posteriors (journal note n0140). Classification coverage per dataset is also unchanged (BBBP 0.54 under both schedules).'))
    b.append("""<p>The conformal coverage failure does not depend on the schedule (registered hypothesis: coverage stays below 0.88; supported), while the Laplace NLL gain does (Table 2d). One backbone, three seeds.</p>""")
    b.append('<h3>8.5 Still open</h3>')
    b.append("""<div class="box warn"><ul><li>Fresh-data confirmation; the schedule test for DNN, GIN and Uni-Mol (only ChemBERTa and, for the bridge, GIN classification were tested); Uni-Mol in the few-label and weighted conformal analyses; heteroscedastic Laplace (not run). The evidence-based prior and Laplace on every ensemble member are done (section 9).</li>
<li>The interpretation of section 8.1 (that the Laplace gain is mostly generic variance inflation) rests on one baseline family fitted on the same validation data; it does not show that no molecule-specific uncertainty is present.</li></ul></div>""")

    b.append('<h2>9. Round 2 (14 October): an evidence-based prior, and Laplace on every ensemble member</h2>')
    b.append("""<p>Registered before the runs (journal n0145, protocol sha256 b7961f11&hellip;); same saved networks as in section 8; exploratory.</p>""")
    b.append('<h3>9.1 Choosing the prior strength by the marginal likelihood (evidence)</h3>')
    b.append("""<p>The prior strength &alpha; was so far chosen per task on validation NLL from a grid and often sat at an edge. A validation-free alternative is the Laplace marginal likelihood: up to terms that do not depend on &alpha;, E(&alpha;) = (D/2) log &alpha; &minus; (&alpha;/2)&Vert;&theta;&Vert;<sup>2</sup> &minus; (1/2) &Sigma;<sub>k</sub> log(&lambda;<sub>k</sub> + &alpha;), where &theta; are the D trained last-layer weights and &lambda;<sub>k</sub> the eigenvalues of the data Hessian on the training set. &alpha; was taken as the maximiser over 10<sup>&minus;8</sup> to 10<sup>4</sup>; no validation or test data enter the choice.</p>""")
    b.append(table(['Backbone', 'Evidence improves vs Deterministic (of 8)', 'Mean NLL change, evidence', 'Mean NLL change, validation-tuned', 'Mean evidence minus validation-tuned', 'Median shift of log<sub>10</sub> &alpha;', 'Datasets where evidence is worse (of 8)'], [
        ['DNN', '7', '&minus;0.009', '&minus;0.077', '+0.068', '+4.2', '6'], ['ChemBERTa', '7', '&minus;0.044', '&minus;0.126', '+0.082', '+3.0', '4'],
        ['GIN', '4', '&minus;0.109', '&minus;0.144', '+0.036', '+3.1', '2'], ['Uni-Mol', '6', '&minus;0.051', '&minus;0.221', '+0.171', '+1.75', '6'],
    ], num_cols=(1, 2, 3, 4, 5, 6), cap='<b>Table 11.</b> Evidence-based against validation-tuned prior strength (journal note n0147; cells c0542, c0543; 12 networks per dataset, Uni-Mol 10). R10 as in section 8; the evidence arm never worsens a dataset. Both registered hypotheses are supported: the evidence &alpha; is larger on 76 to 80 % of task fits with a worse mean NLL by at least 0.02 on all four backbones, and the evidence arm still improves on Deterministic on at least 4 of 8 datasets for every backbone.'))
    b.append("""<ul><li><b>The principled prior gives a much smaller gain.</b> A training-data evidence favours a strong prior (a narrow posterior) because the networks are early-stopped and fit the training data closely; the validation-tuned small &alpha; instead widens the variances until the validation calibration is right. This agrees with section 8.1: what helps is mostly a variance inflation that the validation data decide, not the posterior the evidence describes.</li>
<li><b>The evidence &alpha; never sits at a grid edge</b> (0 % of task fits, against 35 to 75 % for the validation-tuned arm), so the edge problem of section 3 disappears, at the price of a smaller gain. The Laplace posterior still improves NLL on most datasets without using any validation data (7, 7, 4 and 6 of 8), which is the useful, honest version of the method when no validation set is available.</li></ul>""")
    b.append('<h3>9.2 Laplace on every member of an ensemble</h3>')
    b.append("""<p>The ensemble is the ten Deterministic networks of seeds 0 to 9, pooled as in the pipeline (means averaged; regression variances averaged; classification probabilities averaged); it reproduced the stored pooled-ensemble NLL exactly. Arms: E<sub>L</sub> (every member replaced by its own Laplace prediction), E<sub>S</sub> (every member replaced by its recalibrated prediction, section 8.1) and the plain ensemble E<sub>D</sub>. The paired per-molecule NLL difference was bootstrapped over test molecules (2,000 resamples shared by all arms).</p>""")
    b.append(table(['Backbone', 'E<sub>L</sub> better than E<sub>D</sub> / worse (of 8)', 'Mean E<sub>L</sub> minus E<sub>D</sub>', 'Single-network Laplace gain (section 8)', 'Sign-flip p, Holm', 'E<sub>S</sub> better than E<sub>D</sub> (of 8)', 'Mean E<sub>S</sub> minus E<sub>D</sub>', 'Mean E<sub>L</sub> minus E<sub>S</sub>'], [
        ['DNN', '4 / 0', '&minus;0.033', '&minus;0.077', '0.035', '4', '&minus;0.070', '+0.036'], ['ChemBERTa', '5 / 0', '&minus;0.088', '&minus;0.126', '0.035', '4', '&minus;0.095', '+0.007'],
        ['GIN', '2 / 2', '&minus;0.022', '&minus;0.144', '0.68', '3', '&minus;0.005', '&minus;0.017'], ['Uni-Mol', '4 / 0', '&minus;0.135', '&minus;0.221', '0.016', '5', '&minus;0.171', '+0.036'],
    ], num_cols=(1, 2, 3, 4, 5, 6, 7), cap='<b>Table 12.</b> Laplace on every ensemble member (journal note n0148). The registered hypothesis that E<sub>L</sub> improves on E<sub>D</sub> on at least 6 of 8 datasets for at least three backbones is <b>rejected</b> (at most 5); the hypothesis that the ensemble gain is smaller than the single-network gain is <b>supported</b> on all four backbones.'))
    b.append("""<ul><li><b>Laplace still helps a DNN, ChemBERTa or Uni-Mol ensemble, but less</b> (the mean gain is 30 to 57 % smaller than for a single network; 85 % smaller for the GIN): pooling already reduces overconfidence. It does <b>not</b> help the GIN ensemble (2 datasets better, 2 worse).</li>
<li><b>Recalibrating each member is as good or better</b> for three of four backbones (mean E<sub>L</sub> minus E<sub>S</sub> +0.007 to +0.036); only for the GIN is Laplace ahead (&minus;0.017). Even on an ensemble, a single fitted scalar per task matches the Laplace gain.</li>
<li>The pooled ensemble variance is an average of member variances and omits the between-member spread (the published aggregation); the comparisons are conditional on that aggregation.</li></ul>""")
    b.extend(code_section(2))
    b.append('<h2 class="pb">Appendix: where each result is recorded</h2>')
    b.append(table(['Topic', 'Journal references'], [
        ['Laplace, DNN', 'protocol n0059; c0185, c0186, c0187, c0188, c0189, c0190; finding n0060'],
        ['Grid sensitivity', 'registration n0061; c0191, c0192; finding n0062'],
        ['Laplace, ChemBERTa', 'protocol n0064; c0195, c0202, c0203; finding n0065'],
        ['Laplace, GIN', 'protocol n0069; c0251, c0252, c0253; finding n0078'],
        ['Conformal', 'protocol n0082; checks c0257, c0259, c0241, notes n0079, n0081, n0083; coverage c0267, figures c0272; notes n0084, n0089, n0090, n0087, n0088'],
        ['Fix plan', 'registered n0095 (checks A to D, reported in Report 1), n0096 (E to G)'],
        ['Check E (statistics)', 'c0300, c0301; notes n0102, n0103'], ['Check F (shift-aware conformal)', 'c0298; note n0101'],
        ['Recalibration baseline and wide grid (8.1, 8.2)', 'registration n0135 (c0506); jobs c0510, c0511; c0527 to c0529; finding n0143'],
        ['Few-label and weighted conformal (8.3)', 'registration n0138 (c0512); c0513 to c0518; finding n0139'], ['Conformal under the linear schedule (8.4)', 'c0519, c0520; finding n0140'],
        ['Evidence-based prior and Laplace on every ensemble member (9)', 'registration n0145 (c0537); jobs c0539, c0540; c0542, c0543; findings n0147, n0148'],
    ], widths=['26%', '74%']))
    return page('Report 2: Last-Layer Laplace and Split Conformal Prediction on Reproduced MUBen Networks',
                'Does Laplace improve uncertainty at lower cost than ensembles, across datasets, split types and backbones, and do conformal sets keep their 90 % coverage under scaffold shift?',
                'Student project on MUBen (molecular uncertainty benchmark) &middot; sc-hub project <code>muben-laplace</code> &middot; status 14 October 2026 &middot; companion: Report 1 (reproduction)',
                '\n'.join(b))


if __name__ == '__main__':
    (HERE / 'Report1.html').write_text(report1(), encoding='utf-8')
    (HERE / 'Report2.html').write_text(report2(), encoding='utf-8')
    print('wrote', HERE / 'Report1.html', HERE / 'Report2.html')
