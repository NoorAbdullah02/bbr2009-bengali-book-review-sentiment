"""Ensemble of BanglaBook runs; ensemble members and class offsets are chosen on VALIDATION only; test is evaluated once."""
import itertools, json, os
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
D = '.'; R = f'{D}/results'
vy = pd.read_csv(f'{D}/data/banglabook/validation.csv').label.values; ty = pd.read_csv(f'{D}/data/banglabook/test.csv').label.values
tags = [t for t in ['v4a', 'v4b', 'v4c', 'v4d', 'v4e', 'v4f'] if os.path.exists(f'{R}/bb_{t}_val.npy')]
V = {t: np.load(f'{R}/bb_{t}_val.npy') for t in tags}; T = {t: np.load(f'{R}/bb_{t}_test.npy') for t in tags}
wf = lambda p, y: f1_score(y, p.argmax(1), average='weighted')
print('single (val wF1 / test wF1):'); [print(t, round(wf(V[t], vy), 4), round(wf(T[t], ty), 4)) for t in tags]
best = None
for k in range(1, len(tags) + 1):
    for sub in itertools.combinations(tags, k):
        s = wf(np.mean([V[t] for t in sub], 0), vy)
        if best is None or s > best[0]: best = (s, sub)
sub = best[1]; pv = np.mean([V[t] for t in sub], 0); pt = np.mean([T[t] for t in sub], 0)
print('chosen members (by val):', sub, 'val wF1', round(best[0], 4))
g = np.linspace(-1.5, 1.5, 13); bo = (wf(pv, vy), (0, 0))
for o0, o1 in itertools.product(g, g):
    s = wf(np.log(pv + 1e-9) + np.array([o0, o1, 0]), vy)
    if s > bo[0]: bo = (s, (o0, o1))
off = np.array([bo[1][0], bo[1][1], 0]); print('class offsets (val-tuned):', bo[1], 'val wF1', round(bo[0], 4))
fin = np.log(pt + 1e-9) + off
res = dict(members=list(sub), offsets=bo[1], val_wf1=bo[0], test=dict(acc=accuracy_score(ty, fin.argmax(1)), wf1=wf(fin, ty), mf1=f1_score(ty, fin.argmax(1), average='macro'),
           wf1_no_offset=wf(pt, ty)))
print(json.dumps(res, indent=1)); print(classification_report(ty, fin.argmax(1), digits=4))
json.dump(res, open(f'{R}/bb_ensemble.json', 'w'), indent=1)
