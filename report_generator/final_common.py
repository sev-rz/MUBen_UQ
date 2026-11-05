"""Shared pieces of the final reports: style, verdict boxes, front matter, code map (imports the table and chart helpers of build_reports)."""
import re
from build_reports import table, grouped_bars, CSS, esc, C, LAP, DS

LATEX_CSS = """
@page { size: A4; margin: 22mm 22mm 22mm 22mm; }
* { box-sizing: border-box; }
body { font-family: 'STIX Two Text', 'Latin Modern Roman', 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.36; color: #000; text-align: justify; hyphens: auto; }
h1 { font-size: 19pt; font-weight: 400; text-align: center; line-height: 1.2; margin: 14pt 0 10pt 0; color: #000; }
.sub { text-align: center; font-size: 11.5pt; margin-bottom: 8pt; font-style: italic; }
.meta { text-align: center; font-size: 10pt; margin-bottom: 14pt; color: #000; }
h2 { font-size: 13.5pt; font-weight: 700; margin: 17pt 0 6pt 0; color: #000; page-break-after: avoid; text-align: left; }
h3 { font-size: 11.2pt; font-weight: 700; margin: 11pt 0 3pt 0; color: #000; page-break-after: avoid; text-align: left; }
p { margin: 0 0 6pt 0; } ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; } li { margin-bottom: 2.5pt; }
.box { border: none; border-top: 0.6pt solid #000; border-bottom: 0.6pt solid #000; background: none; padding: 4pt 0; margin: 6pt 0 10pt 0; border-radius: 0; }
.box h3 { margin-top: 2pt; } .warn { background: none; }
.contents { margin: 4pt 0 14pt 0; text-align: left; font-size: 10.4pt; } .contents .ct { font-weight: 700; font-size: 12pt; margin-bottom: 3pt; } .contents div { margin: 1.5pt 0; }
.toc { columns: 2; column-gap: 24pt; font-size: 9.6pt; text-align: left; } .toc div { margin: 1pt 0; } .toc .s { margin-left: 12pt; }
table { border-collapse: collapse; width: auto; min-width: 62%; margin: 2pt auto 4pt auto; font-size: 9.3pt; border-top: 1.2pt solid #000; border-bottom: 1.2pt solid #000; text-align: left; }
table.wide { width: 100%; } table, .tcap, th, td { hyphens: none; } .tw { page-break-inside: avoid; break-inside: avoid; } .tw.long { page-break-inside: auto; break-inside: auto; }
th { background: none; color: #000; font-weight: 400; text-align: left; padding: 3pt 7pt; border-bottom: 0.6pt solid #000; vertical-align: bottom; }
td { padding: 1.8pt 7pt; border: none; vertical-align: top; } tr:nth-child(even) td { background: none; }
tr { page-break-inside: avoid; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; } td.c, th.c { text-align: center; }
tr.mid td { border-top: 0.6pt solid #000; }
.cap { font-size: 9.4pt; margin: 3pt 8pt 11pt 8pt; text-align: justify; page-break-inside: avoid; } .cap b { font-weight: 700; color: #000; } .cap i { font-style: italic; }
.tcap { font-size: 9.4pt; margin: 8pt 8pt 3pt 8pt; text-align: justify; page-break-after: avoid; } .tcap b { font-weight: 700; }
figure { margin: 8pt 0 4pt 0; text-align: center; page-break-inside: avoid; } figure img { width: 80%; height: auto; } figure.w img { width: 92%; }
.small { font-size: 9.2pt; }
.pill { display: inline; padding: 0; border-radius: 0; font-size: inherit; color: #000; font-style: italic; } .pill.ok { color: #1B7F8E; } .pill.bad { color: #C9622B; } .pill.mid { color: #14283C; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: none; padding: 0; }
.pb { page-break-before: always; } .katex { font-size: 1.04em; } .katex-display { margin: 5pt 0 6pt 0; page-break-inside: avoid; }
.verdict { border-left: 2.5pt solid #14283C; background: none; padding: 1pt 0 1pt 9pt; margin: 6pt 0 10pt 0; page-break-inside: avoid; text-align: left; }
.verdict.ok { border-color: #1B7F8E; } .verdict.no { border-color: #C9622B; } .verdict.mid { border-color: #A9B0B6; }
.hyp { border-left: 2.5pt solid #9CCBD2; background: none; border-radius: 0; border-top: none; border-right: none; border-bottom: none; padding: 1pt 0 1pt 9pt; margin: 4pt 0 7pt 0; page-break-inside: avoid; text-align: left; }
.hyp b.k { color: #000; font-style: italic; }
"""


def verdict(kind, text):
    """kind: ok | no | mid"""
    label = {'ok': 'Verdict: supported', 'no': 'Verdict: not supported', 'mid': 'Verdict: partly supported'}[kind]
    return f'<div class="verdict {kind}"><b>{label}.</b> {text}</div>'


def hyp(text):
    if text[:1].islower():
        text = text[0].upper() + text[1:]
    return f'<div class="hyp"><b class="k">Hypothesis.</b> {text}</div>'


def toc_from(body):
    """Contents: main sections only, one column, no box (LaTeX \\tableofcontents look)."""
    items = [re.sub('<.*?>', '', m.group(2)).split(', ')[0] for m in re.finditer(r'<h([23])[^>]*>(.*?)</h\1>', body) if m.group(1) == '2' or m.group(2) in ('Goals', 'Summary')]
    return '<div class="contents"><div class="ct">Contents</div>' + ''.join(f'<div>{t}</div>' for t in items) + '</div>'


def table(header, rows, num_cols=(), widths=None, cap=None, wide=False, center_cols=(), rules=()):
    """booktabs-style table, caption above (as in LaTeX). Caption text may start with <b>Table N.</b>; numbering is redone in page()."""
    out = [f'<div class="tw{" long" if len(rows) > 14 else ""}">']
    if cap:
        out.append(f'<div class="tcap">{cap}</div>')
    out.append('<table' + (' class="wide"' if wide or widths or len(header) >= 6 else '') + '>')
    if widths:
        out.append('<colgroup>' + ''.join(f'<col style="width:{w}">' for w in widths) + '</colgroup>')
    cls = lambda i: 'n' if i in num_cols else ('c' if i in center_cols else '')
    out.append('<thead><tr>' + ''.join(f'<th class="{cls(i)}">{h}</th>' for i, h in enumerate(header)) + '</tr></thead><tbody>')
    for ri, r in enumerate(rows):
        out.append(f'<tr class="{"mid" if ri in rules else ""}">' + ''.join(f'<td class="{cls(i)}">{c}</td>' for i, c in enumerate(r)) + '</tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def fig(path, cap, wide=False):
    return f'<figure{" class=w" if wide else ""}><img src="{path}"><div class="cap">{cap}</div></figure>'


def number_captions(body):
    """Renumber '<b>Table N.</b>' and '<b>Figure N.</b>' in order of appearance, LaTeX style 'Table N:'."""
    cnt = {'Table': 0, 'Figure': 0}

    def sub(m):
        cnt[m.group(1)] += 1
        return f'<b>{m.group(1)} {cnt[m.group(1)]}:</b>'
    return re.sub(r'<b>(Table|Figure) \d+\.</b>', sub, body)


def page(title, sub, meta, body):
    body = number_captions(body)
    toc = toc_from(body)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title><link rel="stylesheet" href="katex/katex.min.css"><style>{LATEX_CSS}</style><script src="katex/katex.min.js"></script><script src="katex/auto-render.min.js"></script></head><body>'
            f'<h1>{title}</h1><div class="sub">{sub}</div>{toc}{body}<script>renderMathInElement(document.body,{{delimiters:[{{left:"$$",right:"$$",display:true}},{{left:"$",right:"$",display:false}}],throwOnError:false}});</script></body></html>')


# ---- tables of the Overleaf document (booktabs layout, same numbers) -----------------------------------------
TEAL, ORANGE = '27,127,142', '201,98,43'


def heat(pct, bold):
    a = {100: .75, 92: .68, 90: .67, 80: .59, 75: .55, 67: .49, 58: .42, 33: .45}[pct]
    col = ORANGE if pct == 33 else TEAL
    t = f'{pct}&thinsp;%'
    return f'<td class="c" style="background:rgba({col},{a})">{"<b>"+t+"</b>" if bold else t}</td>'


def consistency_table(cap):
    H = [('DNN', [(100, 1), (100, 1), (58, 0), (100, 1), (92, 1), (100, 1), (100, 1), (100, 1)]),
         ('ChemBERTa', [(100, 1), (100, 0), (33, 0), (100, 1), (100, 1), (92, 1), (100, 1), (92, 1)]),
         ('GIN', [(92, 1), (100, 1), (92, 1), (100, 1), (75, 0), (75, 0), (58, 0), (67, 0)]),
         ('Uni-Mol', [(100, 1), (100, 1), (80, 0), (100, 1), (100, 1), (100, 1), (100, 1), (90, 0)])]
    out = [f'<div class="tw"><div class="tcap">{cap}</div><table class="wide"><thead><tr><th>Backbone</th>' + ''.join(f'<th class="c">{d}</th>' for d in DS) + '</tr></thead><tbody>']
    for name, cells in H:
        out.append(f'<tr><td>{name}</td>' + ''.join(heat(p, b) for p, b in cells) + '</tr>')
    out.append('</tbody></table></div>')
    return '\n'.join(out)


CODE_MAP = [
    ['<code>expanded_v4</code>', 'DNN and ChemBERTa pipeline (training units, evaluation behind FROZEN markers) and comparison with the published tables', 'Report 1'],
    ['<code>expanded_v8_gin</code>, <code>expanded_v10_unimol</code>', 'GIN and Uni-Mol with the authors&rsquo; encoders; their Laplace stages and controls', 'Reports 1 and 2'],
    ['<code>expanded_v9_checks</code>', 'bridge analysis against the authors&rsquo; trainers, descriptor drift, random-split identity, statistics, shift-aware conformal check', 'Reports 1 and 2'],
    ['<code>expanded_v11</code>, <code>v12</code>, <code>v17</code>', 'learning-rate schedule and head-initialisation tests for ChemBERTa and GIN', 'Report 1'],
    ['<code>expanded_v6_laplace</code>, <code>v13</code>, <code>v14</code>, <code>v15</code>, <code>v18</code>', 'last-layer Laplace; schedule test; ten-network analysis; recalibration baseline and wide grid; evidence-based prior and ensembles', 'Report 2'],
    ['<code>expanded_v5_conformal</code>, <code>v16_round1</code>', 'split conformal prediction; few-label and weighted calibration; conformal under the linear schedule', 'Report 2'],
]
