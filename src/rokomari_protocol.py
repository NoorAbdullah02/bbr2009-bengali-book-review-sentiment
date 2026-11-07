"""Rokomari, protocol of Sarker et al. 2022 (pos/neg, <=300 words, 80/10/10 stratified). Their shuffle is unseeded, so the exact split is not reproducible; 3 shuffles."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import cv_bert as cb
d = pd.read_excel(f'{cb.D}/data/rokomari/annotated_reviews.xlsx')
d = d[d.category != 'neutral']
d = d[d.review.astype(str).apply(lambda x: len(x.split()) <= 300)]
T = [cb.clean(t) for t in d.review]; Y = (d.category == 'positive').astype(int).values
print(len(T), np.bincount(Y), flush=True)
res = []
for s in [0, 1, 2]:
    a, tt, ya, yt = train_test_split(T, Y, test_size=0.2, random_state=12342 + s, stratify=Y)
    v, te, yv, ye = train_test_split(tt, yt, test_size=0.5, random_state=12342 + s, stratify=yt)
    m, tok = cb.train(a, ya, s, 4)
    p = cb.probs(m, tok, te).argmax(1)
    r = dict(seed=s, acc=accuracy_score(ye, p), weighted_f1=f1_score(ye, p, average='weighted'), macro_f1=f1_score(ye, p, average='macro'), n_test=len(te))
    print(r, flush=True); res.append(r)
    json.dump(res, open(f'{cb.D}/results/rokomari_protocol.json', 'w'), indent=1)
