"""Redraw results/active_learning/discovery_curves.png from curves_per_seed.csv.

Validated categorical palette (dataviz validator: light mode, CVD/normal-vision checks pass;
contrast WARN mitigated by legend, direct end labels and summary.csv as the table view).
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'results'/'active_learning'
ORDER=['random','greedy','entropy','diversity','balanced'];COLOR=dict(zip(ORDER,['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']))
rows=list(csv.DictReader((OUT/'curves_per_seed.csv').open()));ends={r['seed']:r for r in csv.DictReader((OUT/'per_seed_endpoints.csv').open()) if r['strategy']=='random'}
seeds=sorted({r['seed'] for r in rows},key=int);n=np.arange(7)*25
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#8a8a85','axes.labelcolor':'#3b3b38','xtick.color':'#5f5f5a','ytick.color':'#5f5f5a'})
fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
for ax,k,title in [(axes[0],'cumulative_positives','Cumulative actives among acquired molecules'),(axes[1],'new_active_scaffolds','New active Murcko scaffolds (absent from initial + validation actives)')]:
    finals=[]
    for st in ORDER:
        M=np.asarray([[float(r[k]) for r in rows if r['seed']==s and r['strategy']==st] for s in seeds])
        for line in M:ax.plot(n,line,color=COLOR[st],alpha=.22,lw=.8)
        ax.plot(n,M.mean(0),color=COLOR[st],lw=2,marker='o',ms=4.5,mec='white',mew=1,label=st);finals.append((M.mean(0)[-1],st))
    if k=='cumulative_positives':
        exp=np.mean([float(r['random_expectation_150']) for r in ends.values()])
        ax.plot(n,n*exp/150,ls=':',color='#5f5f5a',lw=1.2,label='random expectation (pool prevalence)')
    # Direct end labels in text ink, nudged apart; color stays on the marks.
    finals.sort();ylo,yhi=ax.get_ylim();gap=(yhi-ylo)*0.055;placed=[]
    for v,st in finals:
        y=max(v,placed[-1]+gap) if placed else v;placed.append(y);ax.annotate(st,(n[-1],v),xytext=(n[-1]+6,y),fontsize=8,color='#3b3b38',va='center',annotation_clip=False)
    ax.set_xlim(-5,n[-1]+38);ax.set_xticks(n);ax.set_xlabel('Acquired labels (beyond 100 initial + 151 validation)');ax.set_title(title,fontsize=9,loc='left',color='#3b3b38')
    ax.grid(axis='y',color='#e6e6e1',lw=.6);ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle(f'BACE retrospective fixed-budget acquisition, {len(seeds)} seeds (thin: per seed; thick: mean)',fontsize=10,color='#1f1f1d')
fig.savefig(OUT/'discovery_curves.png',dpi=160);print('saved',OUT/'discovery_curves.png')
