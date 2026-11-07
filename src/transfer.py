"""Cross-dataset transfer: train on one source, test on the others (BanglaBERT base, 4 epochs, 1 seed). Overlapping texts removed from training."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import cv_bert as cb
mT, mY = cb.load('mendeley'); rT, rY = cb.load('rokomari')
p = pd.read_csv(f'{cb.D}/data/processed/primary_2009.csv').drop_duplicates('text')
b = p[p.source == 'banglabook']; bT, bY = [cb.clean(t) for t in b.text], b.label.values
ms = set(mT) | set(bT)
keep = [i for i, t in enumerate(rT) if t not in ms]
rT, rY = [rT[i] for i in keep], rY[keep]
D = {'mendeley': (mT, mY), 'rokomari': (rT, rY), 'banglabook_subset': (bT, bY)}
print({k: (len(v[0]), np.bincount(v[1]).tolist()) for k, v in D.items()}, flush=True)
res = []
for a in ['mendeley', 'rokomari']:
    m, tok = cb.train(*D[a], 0, 4)
    for t in D:
        if t == a: continue
        T, Y = D[t]
        pr = cb.probs(m, tok, T).argmax(1)
        r = dict(train=a, test=t, n=len(T), acc=accuracy_score(Y, pr), macro_f1=f1_score(Y, pr, average='macro'))
        print(r, flush=True); res.append(r)
    del m
json.dump(res, open(f'{cb.D}/results/transfer.json', 'w'), indent=1)
