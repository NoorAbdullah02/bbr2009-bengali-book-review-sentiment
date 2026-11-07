"""Explainability for BanglaBERT on the primary dataset: one 80/20 stratified split, SHAP token attributions,
global top-token bar charts, error analysis."""
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import cv_bert as cb

plt.rcParams['font.family'] = 'Noto Sans Bengali'
D, OUT = cb.D, f'{cb.D}/results/explain'
import os
os.makedirs(OUT, exist_ok=True)

T, Y = cb.load('primary')
tr_t, te_t, tr_y, te_y = train_test_split(T, Y, test_size=0.2, stratify=Y, random_state=42)
model, tok = cb.train(tr_t, tr_y, 42, 4)
P = cb.probs(model, tok, te_t)
pred = P.argmax(1)
rep = classification_report(te_y, pred, target_names=['negative', 'positive'], digits=4)
open(f'{OUT}/classification_report.txt', 'w').write(rep)
print(rep)

# error analysis
err = pd.DataFrame({'text': te_t, 'true': te_y, 'pred': pred, 'p_pos': P[:, 1]})
err[err.true != err.pred].sort_values('p_pos').to_csv(f'{OUT}/misclassified.csv', index=False)
print('errors', int((err.true != err.pred).sum()), 'of', len(err))


def f(texts):
    return cb.probs(model, tok, [str(t) for t in texts])


masker = shap.maskers.Text(tok)
expl = shap.Explainer(f, masker, output_names=['negative', 'positive'])
n = 60
rng = np.random.RandomState(0)
sel = rng.choice(len(te_t), n, replace=False)
sv = expl([te_t[i] for i in sel])
html = shap.plots.text(sv[:, :, 1], display=False)
open(f'{OUT}/shap_examples.html', 'w').write(html)

# global: mean signed SHAP of words toward "positive"
agg, cnt = defaultdict(float), defaultdict(int)
for e in sv[:, :, 1]:
    for tkn, v in zip(e.data, e.values):
        t = str(tkn).strip()
        if len(t) > 1 and not t.startswith('['):
            agg[t] += v
            cnt[t] += 1
rows = pd.DataFrame([(t, agg[t] / cnt[t], cnt[t]) for t in agg], columns=['token', 'mean_shap', 'count'])
rows = rows[rows['count'] >= 2].sort_values('mean_shap')
rows.to_csv(f'{OUT}/token_shap.csv', index=False)
top = pd.concat([rows.head(12), rows.tail(12)])
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top.token, top.mean_shap, color=['#c0392b' if v < 0 else '#2874a6' for v in top.mean_shap])
ax.set_xlabel('mean SHAP toward positive class')
ax.set_title('Most influential tokens (BanglaBERT)')
plt.tight_layout()
plt.savefig(f'{OUT}/top_tokens.png', dpi=200)
print('done')
