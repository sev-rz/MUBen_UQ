"""Report figures in the style of the Overleaf document (serif text, boxed axes, light grid, teal/orange palette). Data: same coordinates as main.tex."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10, 'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': .6,
                     'axes.axisbelow': True, 'axes.linewidth': .8, 'legend.frameon': False, 'savefig.dpi': 220, 'figure.facecolor': 'white'})
T, O, N, G, LT, B = '#1B7F8E', '#C9622B', '#14283C', '#A9B0B6', '#9CCBD2', '#6C8EAD'
DS = ['BACE', 'BBBP', 'ClinTox', 'Tox21', 'ESOL', 'FreeSolv', 'Lipo', 'QM7']


def save(fig, name):
    fig.savefig(f'figs4/{name}.png', bbox_inches='tight', pad_inches=.05); plt.close(fig)


def hbars(name, labels, series, colors, legend, xlim, xlabel, vline=None, fmt='{:g}', h=3.6, w=6.6, leg_cols=3):
    fig, ax = plt.subplots(figsize=(w, h)); n = len(series); bh = .8 / n
    y = np.arange(len(labels))
    for k, (s, c) in enumerate(zip(series, colors)):
        yy = y - .4 + bh * (k + .5)
        v = [np.nan if x is None else x for x in s]
        ax.barh(yy, v, bh * .92, color=c, left=0)
        for yi, x in zip(yy, v):
            if not np.isnan(x): ax.text(x + (xlim[1] - xlim[0]) * .008, yi, fmt.format(x), va='center', fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(*xlim); ax.set_xlabel(xlabel); ax.grid(axis='y', visible=False)
    if vline: ax.axvline(vline, color='k', ls='--', lw=1)
    if legend:
        from matplotlib.patches import Patch
        ax.legend([Patch(color=c) for c in colors], legend, loc='upper center', bbox_to_anchor=(.5, -.2 if h > 3 else -.3), ncol=leg_cols, fontsize=8)
    save(fig, name)


def vbars(name, groups, series, colors, legend, ylabel, ylim=None, w=6.8, h=3.0, bw=None, top=True, errs=None, labels=False, fmt='{:.2f}', xticks=None):
    fig, ax = plt.subplots(figsize=(w, h)); n = len(series); x = np.arange(len(groups)); bw = bw or .8 / n
    for k, (s, c) in enumerate(zip(series, colors)):
        pos = x - .4 + bw * (k + .5) if n > 1 else x
        ax.bar(pos, s, bw * .95, color=c, yerr=None if errs is None else errs[k], capsize=2.5, error_kw={'lw': .8})
        if labels:
            for p, v, e in zip(pos, s, [0] * len(s) if errs is None else errs[k]): ax.text(p, v + e + (ylim[1] - ylim[0]) * .01, fmt.format(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5); ax.set_ylabel(ylabel); ax.grid(axis='x', visible=False)
    if ylim: ax.set_ylim(*ylim)
    if top: ax.xaxis.tick_top(); ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    if legend:
        from matplotlib.patches import Patch
        ax.legend([Patch(color=c) for c in colors], legend, loc='upper center', bbox_to_anchor=(.5, -.04 if top else -.12), ncol=len(legend), fontsize=8.5)
    save(fig, name)


rows = ['DNN scaffold', 'DNN random', 'ChemBERTa scaffold', 'ChemBERTa random', 'GIN scaffold', 'Uni-Mol scaffold']
hbars('fig1_agreement', rows, [[46.0, 57.6, 37.5, 69.4, 36.4, 52.0], [74.6, 78.8, 65.3, 88.2, 70.3, 75.8]], [T, LT], ['within 1 SD', 'within 2 SD'], (0, 100), '% of published rows matched', h=3.9, leg_cols=2)
vbars('fig2_schedule_gap', ["Authors' linear", "Authors' constant", 'Ours constant', 'Ours linear'], [[0.764, 1.037, 1.033, 0.758]], [O], None, 'test NLL (lower is better)', (0, 1.25), w=5.2, h=3.0, top=False, errs=[[.054, .082, .027, .051]], labels=True, bw=.45)
# colour per bar for fig2
import matplotlib.pyplot as plt2
fig, ax = plt.subplots(figsize=(5.2, 3.0)); xs = np.arange(4); vals = [0.764, 1.037, 1.033, 0.758]; er = [.054, .082, .027, .051]
ax.bar(xs, vals, .45, color=[O, T, T, O], yerr=er, capsize=2.5, error_kw={'lw': .8}); ax.set_ylim(0, 1.25); ax.set_ylabel('test NLL (lower is better)'); ax.grid(axis='x', visible=False)
ax.set_xticks(xs); ax.set_xticklabels(["Authors' linear", "Authors' constant", 'Ours constant', 'Ours linear'], fontsize=8.5)
for i, (v, e) in enumerate(zip(vals, er)): ax.text(i, v + e + .015, f'{v:.2f}', ha='center', fontsize=8)
save(fig, 'fig2_schedule_gap')
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
for ax, t, vs, es, yl in ((axs[0], 'RMSE', [142.9, 126.9, 127.3], [2.9, 0, 0], 175), (axs[1], 'NLL', [10.30, 5.47, 5.48], [1.08, 0, 0], 13)):
    ax.bar(range(3), vs, .45, color=[G, O, T], yerr=es, capsize=2.5, error_kw={'lw': .8}); ax.set_ylim(0, yl); ax.set_title(t, fontsize=10); ax.grid(axis='x', visible=False)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Published', "Authors' code", 'Ours'], fontsize=8.5)
    for i, (v, e) in enumerate(zip(vs, es)): ax.text(i, v + e + yl * .012, f'{v:g}', ha='center', fontsize=8)
fig.tight_layout(); save(fig, 'fig3_qm7')
vbars('fig4_laplace_gain', DS, [[-0.0992, -0.0346, -0.0, -0.0033, -0.167, -0.2365, -0.04, -0.0358], [-0.1012, -0.0522, -0.0025, -0.0115, -0.4413, -0.2243, -0.1182, -0.0533],
      [-0.0237, -0.1881, -0.0461, -0.0399, -0.2506, -0.5769, -0.0231, -0.0057], [-0.2534, -0.47, -0.0048, -0.0163, -0.387, -0.3826, -0.2247, -0.029]], [B, T, O, N],
      ['DNN', 'ChemBERTa', 'GIN', 'Uni-Mol'], r'$\Delta$NLL (Laplace $-$ Deterministic)', (-0.65, 0.02), h=3.2)
vbars('fig5_schedule_gain', DS, [[-0.098, -0.104, 0, -0.014, -0.476, -0.196, -0.116, -0.079], [-0.084, -0.033, 0, -0.007, -0.157, -0.096, -0.055, -0.01]], [T, O],
      ['constant schedule', 'linear schedule, 10% warm-up'], r'$\Delta$NLL (Laplace $-$ Deterministic)', (-0.55, 0.02), h=3.0)
vbars('fig6_unimol_control', DS, [[0.0936, 0.1119, 0.0002, 0.003, 0.0014, 0.001, 0.041, 0.0003], [-0.378, -0.6972, -0.0048, -0.0191, -0.4176, -0.4494, -0.2496, -0.0219]], [O, T],
      ['averaging conformers only', 'Laplace posterior'], 'change in test NLL', (-0.78, 0.16), h=3.0)
hbars('fig7_coverage', ['DNN', 'ChemBERTa', 'GIN', 'Uni-Mol'], [[.9, .91, .886, None], [.788, .819, .753, .816], [.904, .904, .904, .905]], [G, O, T],
      ['random split', 'scaffold split', 'scaffold calibrated on target scaffolds'], (0.6, 1.0), 'coverage at a 0.90 target', vline=.9, fmt='{:.2f}', h=3.5, w=6.6)
m = ['Ensembles', 'Deterministic', 'MC dropout', 'Temperature', 'BBP', 'SWAG', 'SGLD', 'Laplace constant', 'Novelty only', 'Novelty Laplace', 'Laplace', 'Focal loss']
v = [.971, 1.0, 1.0, 1.008, 1.01, 1.011, 1.015, 1.032, 1.036, 1.042, 1.048, 1.06]
hbars('fig8_setsize', m, [v], [O], None, (0.95, 1.08), 'set size relative to Deterministic', vline=1.0, fmt='{:g}', h=4.0, w=6.2)
