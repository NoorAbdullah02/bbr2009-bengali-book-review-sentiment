"""Extended explainability for BanglaBERT on the primary set.
1) SHAP on ALL 402 held-out reviews of one 80/20 split (token level, global top tokens per class)
2) Faithfulness: deletion test - remove top-k SHAP tokens vs k random tokens, mean drop in predicted-class probability
3) SHAP on the 69 suspected-noisy labels (results/suspect_labels.csv): which words drive the model against the given label
4) Per-source (Mendeley vs BanglaBook) top tokens."""
import os, json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
import cv_bert as cb

plt.rcParams['font.family'] = 'Noto Sans Bengali'
OUT = f'{cb.D}/results/explain2'
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(f'{cb.D}/data/processed/primary_2009.csv').drop_duplicates('text').reset_index(drop=True)
T = [cb.clean(t) for t in d.text]; Y = d.label.values; S = d.source.values
idx = np.arange(len(T))
tr, te = train_test_split(idx, test_size=0.2, stratify=Y, random_state=42)
model, tok = cb.train([T[i] for i in tr], Y[tr], 42, 4)
f = lambda x: cb.probs(model, tok, [str(t) for t in x])
expl = shap.Explainer(f, shap.maskers.Text(tok), output_names=['negative', 'positive'])


def run(ids):
    return expl([T[i] for i in ids], batch_size=16)


def tokstats(sv, sign_class=1):
    agg, cnt = defaultdict(float), defaultdict(int)
    for e in sv:
        for t, v in zip(e.data, e.values[:, sign_class]):
            t = str(t).strip()
            if len(t) > 1 and not t.startswith('['):
                agg[t] += v; cnt[t] += 1
    r = pd.DataFrame([(t, agg[t] / cnt[t], cnt[t]) for t in agg], columns=['token', 'mean_shap', 'count'])
    return r[r['count'] >= 3].sort_values('mean_shap')


def bar(r, path, title):
    top = pd.concat([r.head(12), r.tail(12)])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.token, top.mean_shap, color=['#c0392b' if v < 0 else '#2874a6' for v in top.mean_shap])
    ax.set_xlabel('mean SHAP toward positive class'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


# 1) all test reviews
sv = run(te)
r = tokstats(sv); r.to_csv(f'{OUT}/token_shap_all402.csv', index=False)
bar(r, f'{OUT}/top_tokens_all402.png', f'Most influential tokens (n={len(te)} reviews)')
for src in ['mendeley', 'banglabook']:
    sel = [k for k, i in enumerate(te) if S[i] == src]
    if len(sel) >= 20:
        rs = tokstats([sv[k] for k in sel]); rs.to_csv(f'{OUT}/token_shap_{src}.csv', index=False)
        bar(rs, f'{OUT}/top_tokens_{src}.png', f'{src} (n={len(sel)})')

# 2) faithfulness: deletion test
rng = np.random.RandomState(0)
P0 = f([T[i] for i in te]); pred = P0.argmax(1)
res = {}
for k in [1, 3, 5]:
    shap_txt, rnd_txt = [], []
    for j, e in enumerate(sv):
        toks = [str(t) for t in e.data]; v = e.values[:, pred[j]]
        order = np.argsort(-v)[:k]
        pool = [q for q, t in enumerate(toks) if t.strip()]
        rr = rng.choice(pool, min(k, len(pool)), replace=False)
        shap_txt.append(''.join(t for q, t in enumerate(toks) if q not in set(order)))
        rnd_txt.append(''.join(t for q, t in enumerate(toks) if q not in set(rr)))
    ps = f(shap_txt)[np.arange(len(te)), pred]; pr = f(rnd_txt)[np.arange(len(te)), pred]
    res[k] = dict(shap_drop=float((P0[np.arange(len(te)), pred] - ps).mean()), random_drop=float((P0[np.arange(len(te)), pred] - pr).mean()))
    print(k, res[k], flush=True)
json.dump(res, open(f'{OUT}/deletion_test.json', 'w'), indent=1)

# 3) suspected noisy labels
sus = pd.read_csv(f'{cb.D}/results/suspect_labels.csv')
pos = {t: i for i, t in enumerate(T)}
ids = [pos[cb.clean(t)] for t in sus.text if cb.clean(t) in pos]
sv2 = run(ids)
rows = []
for i, e in zip(ids, sv2):
    top = np.argsort(-np.abs(e.values[:, 1 - Y[i]]))[:5]  # words pushing toward the class the model predicts
    rows.append(dict(text=T[i][:200], given_label=int(Y[i]), source=S[i], p_pos=float(f([T[i]])[0, 1]),
                     key_words=' | '.join(str(e.data[q]).strip() for q in top)))
pd.DataFrame(rows).to_csv(f'{OUT}/suspect_label_words.csv', index=False)
print('done')
