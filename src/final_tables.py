"""Aggregate every result into results/final_tables.md + figures. Ensemble uses saved out-of-fold probabilities."""
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

R = './results'
import sys
sys.path.insert(0, './src')
import cv_bert as cb

L = []
fmt = lambda s: f'{s.mean()*100:.2f} ± {s.std()*100:.2f}'

# --- base / large / ensemble, 3 seeds x 5 folds, rebuilt from saved out-of-fold probabilities ---
_, Y = cb.load('primary')
rows = []
for sd in range(3):
    for f in range(5):
        idx = np.load(f'{R}/bert_primary_s{sd}_f{f}_idx.npy')
        pb = np.load(f'{R}/bert_primary_s{sd}_f{f}_prob.npy')
        pl = np.load(f'{R}/bert_large_primary_s{sd}_f{f}_prob.npy')
        for name, p in [('base', pb), ('large', pl), ('ensemble', (pb + pl) / 2)]:
            pr, rc, f1, _ = precision_recall_fscore_support(Y[idx], p.argmax(1), average='macro')
            rows.append(dict(model=name, seed=sd, fold=f, acc=accuracy_score(Y[idx], p.argmax(1)), f1=f1))
ens = pd.DataFrame(rows)
ens.to_csv(f'{R}/ensemble_primary.csv', index=False)
L.append('## BanglaBERT base / large / ensemble (primary 2,009, 3 seeds x 5-fold = 15 runs)\n')
L.append('| Model | Accuracy % | Macro-F1 % |\n|---|---|---|')
for n in ['base', 'large', 'ensemble']:
    e = ens[ens.model == n]
    L.append(f'| {n} | {fmt(e.acc)} | {fmt(e.f1)} |')
a = ens.pivot_table(index=['seed', 'fold'], columns='model', values='acc')
L.append(f'\nPaired per-run diff ensemble-base: mean {(a.ensemble - a.base).mean()*100:.2f} pts, Wilcoxon p={wilcoxon(a.ensemble, a.base).pvalue:.3f}; '
         f'large-base: mean {(a.large - a.base).mean()*100:.2f} pts, p={wilcoxon(a.large, a.base).pvalue:.3f}\n')

# --- main comparison ---
L.append('## Main results (5-fold CV accuracy %)\n')
L.append('| Dataset | BanglaBERT | Best classical (TF-IDF) | Best from-scratch RNN | Previous best (paper) |\n|---|---|---|---|---|')
rnn = pd.read_csv(f'{R}/cv_rnn_raw.csv').groupby('model').cv_acc.agg(['mean', 'std']).sort_values('mean', ascending=False)
prev = {'mendeley': '84% (NB, Hossain 2020)', 'rokomari': '88.95% wF1 (XLM-R, Sarker 2022)', 'primary': 'n/a'}
for ds in ['mendeley', 'primary', 'rokomari']:
    b = pd.read_csv(f'{R}/cv_bert_{ds}.csv')
    if ds == 'primary':
        b = ens[ens.model == 'base'].rename(columns={'acc': 'cv_acc', 'f1': 'cv_f1'})
    c = pd.read_csv(f'{R}/cv_classical_{ds}.csv').groupby('model').acc.agg(['mean', 'std']).sort_values('mean', ascending=False)
    r = f'{rnn.index[0]} {rnn["mean"].iloc[0]*100:.2f} ± {rnn["std"].iloc[0]*100:.2f}' if ds == 'primary' else 'n/a'
    L.append(f'| {ds} | {fmt(b.cv_acc)} (F1 {fmt(b.cv_f1)}) | {c.index[0]} {c["mean"].iloc[0]*100:.2f} ± {c["std"].iloc[0]*100:.2f} | {r} | {prev[ds]} |')

for ds in ['mendeley', 'primary', 'rokomari']:
    c = pd.read_csv(f'{R}/cv_classical_{ds}.csv').groupby('model')[['acc', 'f1']].agg(['mean', 'std']).sort_values(('acc', 'mean'), ascending=False)
    L.append(f'\n### Classical TF-IDF baselines, {ds}\n\n| Model | Acc % | Macro-F1 % |\n|---|---|---|')
    for m, r in c.iterrows():
        L.append(f'| {m} | {r[("acc","mean")]*100:.2f} ± {r[("acc","std")]*100:.2f} | {r[("f1","mean")]*100:.2f} |')

d = pd.read_csv(f'{R}/cv_rnn_raw.csv')
g = d.groupby('model')[['cv_acc', 'cv_f1', 'ext_acc']].agg(['mean', 'std']).sort_values(('cv_acc', 'mean'), ascending=False)
L.append('\n### From-scratch recurrent models, primary 2,009 (5 seeds × 5-fold = 25 runs each)\n\n| Model | CV Acc % | CV Macro-F1 % | External Acc % |\n|---|---|---|---|')
for m, r in g.iterrows():
    L.append(f'| {m} | {r[("cv_acc","mean")]*100:.2f} ± {r[("cv_acc","std")]*100:.2f} | {r[("cv_f1","mean")]*100:.2f} | {r[("ext_acc","mean")]*100:.2f} |')

L.append('\n### BanglaBERT ablation (primary, seed 0, 5-fold)\n\n| Variant | Acc % |\n|---|---|')
for tag, nm in [('', 'default (LR 3e-5, len 128, 4 ep, class weight)'), ('_abl_lr5e5', 'LR 5e-5'), ('_abl_nocw', 'no class weight'),
                ('_abl_len64', 'max length 64'), ('_abl_ep2', '2 epochs')]:
    e = pd.read_csv(f'{R}/cv_bert{tag}_primary.csv')
    L.append(f'| {nm} | {fmt(e.cv_acc)} |')

# Wilcoxon: BanglaBERT vs best classical on mendeley, per fold (seed 0)
b = pd.read_csv(f'{R}/cv_bert_mendeley.csv')
c = pd.read_csv(f'{R}/cv_classical_mendeley.csv')
best = c.groupby('model').acc.mean().idxmax()
bb = b[b.seed == 0].sort_values('fold').cv_acc.values
cc = c[(c.model == best) & (c.seed == 0)].sort_values('fold').acc.values
L.append(f'\nMendeley seed 0: BanglaBERT per-fold {np.round(bb*100,2).tolist()} vs {best} {np.round(cc*100,2).tolist()}; '
         f'min per-fold gain {np.round((bb-cc).min()*100,2)} pts.')
open(f'{R}/final_tables.md', 'w').write('\n'.join(L))

# --- figures ---
fig, ax = plt.subplots(figsize=(7, 4))
labels, vals, errs = [], [], []
for ds in ['mendeley', 'primary', 'rokomari']:
    b = ens[ens.model == 'base'].acc if ds == 'primary' else pd.read_csv(f'{R}/cv_bert_{ds}.csv').cv_acc
    c = pd.read_csv(f'{R}/cv_classical_{ds}.csv').groupby('model').acc.mean().max()
    labels += [f'{ds}\nBanglaBERT', f'{ds}\nbest classical']
    vals += [b.mean() * 100, c * 100]
    errs += [b.std() * 100, 0]
ax.bar(range(len(vals)), vals, yerr=errs, color=['#2874a6', '#aab7b8'] * 3, capsize=3)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylim(80, 100)
ax.set_ylabel('CV accuracy (%)')
plt.tight_layout()
plt.savefig(f'{R}/fig_main.png', dpi=200)
print('\n'.join(L))
