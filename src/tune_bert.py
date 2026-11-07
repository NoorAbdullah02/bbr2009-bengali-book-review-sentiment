"""Hyper-parameter search for a transformer on BBR-2009 without touching the evaluation folds:
a tuning split = 80/20 of the TRAINING part of fold 0 of seed 0 (the fold-0 test part is never used). usage: BB_MODEL=... python3 tune_bert.py <tag>"""
import json, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import cv_bert as cb
tag = sys.argv[1]
T, Y = cb.load('primary')
tr, _ = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(T, Y)))
a, b = train_test_split(tr, test_size=0.2, stratify=Y[tr], random_state=0)
res = []
for lr in [1e-5, 2e-5, 3e-5, 5e-5]:
    for ep in [4, 6]:
        cb.LR = lr
        m, tok = cb.train([T[i] for i in a], Y[a], 0, ep)
        acc = float((cb.probs(m, tok, [T[i] for i in b]).argmax(1) == Y[b]).mean())
        res.append(dict(lr=lr, epochs=ep, tune_acc=acc)); print(res[-1], flush=True); del m
best = max(res, key=lambda r: r['tune_acc']); print('BEST', best)
json.dump(dict(grid=res, best=best), open(f'./results/tune_{tag}.json', 'w'), indent=1)
