"""Publication figures for the conference paper (print: white background, one hue for 'ours')."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
O = './figures/'
BL, OR, AQ, GR, YE, INK, MUT = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a95', '#eda100', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': MUT, 'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': INK, 'axes.linewidth': 0.7})

# ---------- Fig 1: pipeline + architecture ----------
fig = plt.figure(figsize=(7.2, 3.7)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 720); ax.set_ylim(0, 370); ax.axis('off')
def box(x, y, w, h, t, c, fs=7, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=5', fc=c, ec='#34607a', lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc, linespacing=1.25)
def arr(x1, y1, x2, y2, c='#34607a'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, color=c, lw=1))
ax.text(8, 352, '(a) Experimental pipeline', fontsize=8.5, fontweight='bold', color=INK)
box(8, 290, 92, 46, 'Data\nMendeley 1,414\n+ BanglaBook 595', '#e6f0fb'); arr(100, 313, 114, 313)
box(114, 290, 86, 46, 'Preprocessing\ndedup, normalize', '#e6f0fb'); arr(200, 313, 214, 313)
box(214, 290, 120, 46, 'Models\n19 TF-IDF, 12 RNN,\n4 transformers', '#cde2fb'); arr(334, 313, 348, 313)
box(348, 290, 100, 46, 'Stratified 5-fold CV\n3-5 seeds', '#fde9df'); arr(448, 313, 462, 313)
box(462, 290, 118, 46, 'Evaluation\nacc, F1, McNemar, ECE', '#fde9df'); arr(580, 313, 594, 313)
box(594, 290, 118, 46, 'Analyses\nSHAP, label noise,\ntransfer', '#d9f1e8')
ax.text(8, 262, '(b) BanglaBERT classifier (fine-tuned end-to-end)', fontsize=8.5, fontweight='bold', color=INK)
y0 = 168
box(8, y0, 78, 62, 'Review text\n(Bengali)', '#e6f0fb'); arr(86, y0 + 31, 102, y0 + 31)
box(102, y0, 92, 62, 'WordPiece\ntokenizer\nmax length 128', '#e6f0fb'); arr(194, y0 + 31, 210, y0 + 31)
# tokens
for i, t in enumerate(['[CLS]', 't1', 't2', '...', 'tn', '[SEP]']):
    box(210 + i * 30, y0 + 70, 26, 18, t, '#ffffff', 6)
    arr(223 + i * 30, y0 + 68, 223 + i * 30, y0 + 58, '#9a9a95')
box(210, y0, 176, 56, 'BanglaBERT encoder (ELECTRA)\n12 (base) / 24 (large) layers\nself-attention + feed-forward', '#9ccbe0', 6.4, True)
arr(386, y0 + 31, 402, y0 + 31)
box(402, y0, 70, 62, '[CLS] vector\n768 / 1024-d', '#ffffff', 6.8); arr(472, y0 + 31, 488, y0 + 31)
box(488, y0, 78, 62, 'Dropout\n+ Linear\n(2 classes)', '#cde2fb', 7); arr(566, y0 + 31, 582, y0 + 31)
box(582, y0, 64, 62, 'Softmax\nclass\nprobabilities', '#fde9df', 6.5); arr(646, y0 + 31, 662, y0 + 31)
box(662, y0 + 14, 50, 34, 'Label', '#d9f1e8', 7.5, True)
ax.text(8, 120, '(c) Ensemble and explanation', fontsize=8.5, fontweight='bold', color=INK)
box(8, 44, 110, 56, 'BanglaBERT (base)\nprob. $p_b$', '#cde2fb'); box(8, 4, 110, 34, 'BanglaBERT-large\nprob. $p_l$', '#cde2fb', 7)
arr(118, 72, 150, 60); arr(118, 21, 150, 52)
box(150, 36, 112, 40, 'Average\n$(p_b+p_l)/2$', '#fde9df', 7.5, True); arr(262, 56, 290, 56)
box(290, 36, 100, 40, 'Ensemble\nprediction', '#d9f1e8', 7.5, True)
box(430, 44, 128, 56, 'SHAP text masker\ntoken attributions\n(Partition explainer)', '#d9f1e8', 7); arr(558, 72, 580, 72)
box(580, 44, 132, 56, 'Faithfulness test\ndelete top-k tokens\nvs random tokens', '#d9f1e8', 7)
box(430, 4, 128, 34, 'Confident errors\n$p>0.9$ -> audit', '#fde9df', 7); arr(558, 21, 580, 21)
box(580, 4, 132, 34, 'Blind re-annotation', '#fde9df', 7)
fig.savefig(O + 'fig1_arch.pdf'); fig.savefig(O + 'fig1_arch.png', dpi=300); plt.close()

# ---------- Fig 2: BBR-2009 bar chart ----------
M = [('Ensemble (base+large)', 94.23, 0.82, BL), ('BanglaBERT base', 93.84, 1.10, BL), ('BanglaBERT-large', 93.81, 1.04, BL),
     ('XLM-R (tuned)', 91.31, 1.12, OR), ('Linear SVM (TF-IDF)', 90.19, 1.53, GR), ('BiLSTM, pretrained emb.', 87.61, 1.75, YE), ('mBERT (tuned)', 86.83, 1.54, OR),
     ('BiLSTM+Attn, random emb.', 81.08, 3.75, YE), ('RNN, random emb.', 73.57, 6.20, YE)]
fig, ax = plt.subplots(figsize=(3.45, 2.75))
y = np.arange(len(M))[::-1]
ax.barh(y, [m[1] - 70 for m in M], left=70, xerr=[m[2] for m in M], color=[m[3] for m in M], height=0.62, ecolor=MUT,
        error_kw=dict(lw=0.8, capsize=2), edgecolor='white')
for yi, m in zip(y, M): ax.text(m[1] + m[2] + 0.5, yi, f'{m[1]:.2f}', va='center', fontsize=6.8, color=INK)
ax.set_yticks(y); ax.set_yticklabels([m[0] for m in M], fontsize=6.6); ax.set_xlim(70, 101.5); ax.set_xlabel('Accuracy (%), mean $\\pm$ std (axis from 70)', fontsize=6.8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BL, label='Bengali transformer'), Patch(color=OR, label='Multilingual transformer'), Patch(color=GR, label='TF-IDF'), Patch(color=YE, label='Recurrent')],
          fontsize=6.3, loc='upper center', bbox_to_anchor=(0.3, -0.27), ncol=2, frameon=False)
ax.xaxis.grid(True, lw=0.4, color='#e6e6e3'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(O + 'fig2_bbr_bars.pdf'); fig.savefig(O + 'fig2_bbr_bars.png', dpi=300); plt.close()

# ---------- Fig 3: dataset comparison + BanglaBook progression ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
ds = ['Mendeley', 'BBR-2009', 'Rokomari']; w = 0.26; x = np.arange(3)
bb = [96.72, 93.84, 90.60]; cl = [94.24, 90.19, 88.42]; pub = [84.0, np.nan, 88.95]
for k, (v, c, l) in enumerate([(bb, BL, 'BanglaBERT (ours)'), (cl, GR, 'Best TF-IDF (ours)'), (pub, OR, 'Published best')]):
    a.bar(x + (k - 1) * w, np.nan_to_num(v, nan=0) - 80, bottom=80, width=w - 0.03, color=c, label=l, edgecolor='white')
    for xi, vi in zip(x, v):
        if not np.isnan(vi): a.text(xi + (k - 1) * w, vi + 0.3, f'{vi:.1f}', ha='center', fontsize=6.3)
a.text(1 + w, 80.6, 'n/a', ha='center', fontsize=6, color=MUT)
a.set_xticks(x); a.set_xticklabels(ds); a.set_ylim(80, 100); a.set_ylabel('Accuracy (%), axis starts at 80', fontsize=7)
a.legend(fontsize=6, frameon=False, loc='upper right', ncol=1); a.yaxis.grid(True, lw=0.4, color='#e6e6e3'); a.set_axisbelow(True)
a.set_title('(a) Rokomari published value is weighted F1 on a different split', fontsize=6.5, loc='left', color=MUT)
lab = ['Run 1\n(unweighted,\n2 ep.)', 'Run 2\n(class wt.,\n3 ep.)', 'Final\n(3-model\nensemble)', 'Published\n(Kabir et al.)']; v = [0.892, 0.9187, 0.9274, 0.9331]
cols = [GR, GR, BL, OR]
b.bar(range(4), np.array(v) - 0.88, bottom=0.88, color=cols, width=0.62, edgecolor='white')
for i, vi in enumerate(v): b.text(i, vi + 0.0012, f'{vi:.3f}', ha='center', fontsize=6.8)
b.set_xticks(range(4)); b.set_xticklabels(lab, fontsize=6.2); b.set_ylim(0.88, 0.945); b.set_ylabel('Test weighted F1 (axis starts at 0.88)', fontsize=7)
b.yaxis.grid(True, lw=0.4, color='#e6e6e3'); b.set_axisbelow(True); b.set_title('(b) Official BanglaBook (3-class)', fontsize=6.5, loc='left', color=MUT)
fig.tight_layout(); fig.savefig(O + 'fig3_datasets.pdf'); fig.savefig(O + 'fig3_datasets.png', dpi=300); plt.close()

# ---------- Fig 4: explainability + label noise ----------
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 1.85))
ks = [1, 3, 5]; sh = [0.120, 0.324, 0.444]; rn = [0.024, 0.095, 0.150]; xx = np.arange(3)
a.bar(xx - 0.19, sh, 0.36, color=BL, label='Top-k SHAP tokens', edgecolor='white'); a.bar(xx + 0.19, rn, 0.36, color=GR, label='Random tokens', edgecolor='white')
for xi, s_, r_ in zip(xx, sh, rn): a.text(xi - 0.19, s_ + 0.008, f'{s_:.2f}', ha='center', fontsize=6.3); a.text(xi + 0.19, r_ + 0.008, f'{r_:.2f}', ha='center', fontsize=6.3)
a.set_xticks(xx); a.set_xticklabels([f'k={k}' for k in ks]); a.set_ylabel('Drop in predicted-class prob.', fontsize=7); a.legend(fontsize=6, frameon=False, loc='upper left')
a.set_title('(a) Deletion faithfulness', fontsize=7, loc='left')
b.bar([0, 1], [8.6, 1.3], color=[OR, BL], width=0.55, edgecolor='white')
for i, v_ in enumerate([8.6, 1.3]): b.text(i, v_ + 0.2, f'{v_:.1f}%', ha='center', fontsize=7)
b.set_xticks([0, 1]); b.set_xticklabels(['BanglaBook\n(rating labels)', 'Mendeley\n(manual labels)'], fontsize=6.5); b.set_ylabel('Confident errors ($p>0.9$), % of reviews', fontsize=6.5); b.set_ylim(0, 10.5)
b.set_title('(b) Suspected label errors', fontsize=7, loc='left')
c.bar([0, 1], [96.6, 25.9], color=[BL, OR], width=0.55, edgecolor='white')
for i, v_ in enumerate([96.6, 25.9]): c.text(i, v_ + 1.5, f'{v_:.1f}%', ha='center', fontsize=7)
c.set_xticks([0, 1]); c.set_xticklabels(['Random\n(n=118)', 'Suspected\n(n=54)'], fontsize=6.5); c.set_ylabel('LLM audit agrees with label (%)', fontsize=6.5); c.set_ylim(0, 112)
c.set_title('(c) Blind LLM audit', fontsize=7, loc='left')
for x_ in (a, b, c): x_.yaxis.grid(True, lw=0.4, color='#e6e6e3'); x_.set_axisbelow(True)
fig.tight_layout(); fig.savefig(O + 'fig4_explain_noise.pdf'); fig.savefig(O + 'fig4_explain_noise.png', dpi=300); plt.close()

# ---------- Fig 5: SHAP tokens (Bengali, with English glosses) ----------
G = {'অখ': 'awful (word piece)', 'বিরক্তিকর': 'annoying', 'লাগেনি': "didn't like", 'বাজে': 'bad', 'ফালতু': 'worthless', 'নি': 'not (suffix)',
     'সেরা': 'best', 'দারুণ': 'great', 'চমৎকার': 'excellent', 'অসাধারন': 'extraordinary', 'অসাধারণ': 'extraordinary', 'বেস্ট': 'best'}
t = pd.read_csv('./results/explain2/token_shap_all402.csv')
t = t[t['count'] >= 4].sort_values('mean_shap'); top = pd.concat([t.head(6), t.tail(6)])
plt.rcParams['font.family'] = 'Noto Sans Bengali'
fig, ax = plt.subplots(figsize=(3.45, 2.1))
ax.barh(range(len(top)), top.mean_shap, color=[OR if v < 0 else BL for v in top.mean_shap], height=0.7, edgecolor='white')
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.token, fontsize=7.5); ax.axvline(0, color=MUT, lw=0.6)
ax.set_xlim(-0.2, 0.72)
for k, tok_ in enumerate(top.token):
    ax.text(0.42, k, G.get(tok_, ''), va='center', ha='left', fontsize=6.2, color=MUT, fontfamily='DejaVu Sans')
ax.set_xlabel('Mean SHAP toward positive class', fontsize=7, fontfamily='DejaVu Sans'); ax.xaxis.grid(True, lw=0.4, color='#e6e6e3'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(O + 'fig5_shap.pdf'); fig.savefig(O + 'fig5_shap.png', dpi=300); plt.close()
print('ok'); print(top.token.tolist(), [G.get(x) for x in top.token])

# ---------- Fig 7: transfer heatmap ----------
import matplotlib.colors as mc
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(3.45, 1.9))
cols_ = ['Mendeley', 'Rokomari', 'BanglaBook\nsubset']; rows_ = ['Mendeley', 'Rokomari']
M_ = np.array([[np.nan, 84.5, 77.0], [91.2, np.nan, 82.4]])
cm_ = mc.LinearSegmentedColormap.from_list('b', ['#cde2fb', '#2a78d6', '#104281'])
ax.imshow(np.ma.masked_invalid(M_), cmap=cm_, vmin=70, vmax=95, aspect='auto')
for i in range(2):
    for j in range(3):
        if np.isnan(M_[i, j]): ax.text(j, i, 'same\nsource', ha='center', va='center', fontsize=6, color=MUT)
        else: ax.text(j, i, f'{M_[i, j]:.1f}', ha='center', va='center', fontsize=8, color='white' if M_[i, j] > 80 else INK, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels(cols_, fontsize=7); ax.set_yticks(range(2)); ax.set_yticklabels(rows_, fontsize=7)
ax.set_xlabel('Test source', fontsize=7); ax.set_ylabel('Train source', fontsize=7)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(O + 'fig7_transfer.pdf'); fig.savefig(O + 'fig7_transfer.png', dpi=300); plt.close()
print('fig7 ok')
