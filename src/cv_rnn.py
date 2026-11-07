"""Repeated stratified k-fold CV for 12 recurrent / attention configs on the primary dataset,
plus evaluation of every fold model on the clean external Rokomari set."""
import json
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

D = '.'
OUT = f'{D}/results'
VOCAB, MAXLEN, EMB, HID, BATCH, EPOCHS, PATIENCE = 15000, 150, 128, 128, 32, 20, 4
dev = torch.device('cuda')


def clean(t):
    t = re.sub(r'[^ঀ-৿\s]', ' ', str(t))
    return re.sub(r'\s+', ' ', t).strip()


class Vocab:
    def __init__(self, texts):
        c = Counter(w for t in texts for w in t.split())
        self.w2i = {w: i + 2 for i, (w, _) in enumerate(c.most_common(VOCAB - 2))}

    def enc(self, t):
        ids = [self.w2i.get(w, 1) for w in t.split()][:MAXLEN] or [1]
        return ids

    def batch(self, texts):
        seqs = [self.enc(t) for t in texts]
        L = torch.tensor([len(s) for s in seqs])
        X = torch.zeros(len(seqs), max(L), dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s)
        return X, L


class Net(nn.Module):
    def __init__(self, cell, bi, attn):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB, padding_idx=0)
        self.drop = nn.Dropout(0.3)
        self.rnn = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell](EMB, HID, batch_first=True, bidirectional=bi)
        d = HID * (2 if bi else 1)
        self.attn = nn.Linear(d, 1) if attn else None
        self.fc = nn.Linear(d, 2)
        self.cell, self.bi = cell, bi

    def forward(self, X, L, return_attn=False):
        e = self.drop(self.emb(X))
        pk = nn.utils.rnn.pack_padded_sequence(e, L.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(pk)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=X.size(1))
        if self.attn is not None:
            s = self.attn(torch.tanh(out)).squeeze(-1).masked_fill(X == 0, -1e9)
            a = torch.softmax(s, 1)
            ctx = (a.unsqueeze(-1) * out).sum(1)
        else:
            if self.cell == 'lstm':
                h = h[0]
            ctx = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
            a = None
        logits = self.fc(self.drop(ctx))
        return (logits, a) if return_attn else logits


def predict(model, vocab, texts, bs=256):
    model.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            X, L = vocab.batch(texts[i:i + bs])
            P.append(torch.softmax(model(X.to(dev), L), 1).cpu().numpy())
    return np.concatenate(P)


def fit(cell, bi, attn, tr_t, tr_y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    a_t, v_t, a_y, v_y = train_test_split(tr_t, tr_y, test_size=0.1, stratify=tr_y, random_state=seed)
    vocab = Vocab(a_t)
    model = Net(cell, bi, attn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cw = torch.tensor(len(a_y) / (2 * np.bincount(a_y)), dtype=torch.float, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    best, best_state, bad = 1e9, None, 0
    rng = np.random.RandomState(seed)
    for ep in range(EPOCHS):
        model.train()
        idx = rng.permutation(len(a_t))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            X, L = vocab.batch([a_t[j] for j in b])
            y = torch.tensor([a_y[j] for j in b], device=dev)
            opt.zero_grad()
            lossf(model(X.to(dev), L), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = predict(model, vocab, v_t)
        vl = float(-np.log(p[np.arange(len(v_y)), v_y] + 1e-9).mean())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, vocab


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1)


CONFIGS = {f'{"Bi" if bi else ""}{c.upper()}{"+Attn" if at else ""}': (c, bi, at)
           for at in (False, True) for c in ('rnn', 'gru', 'lstm') for bi in (False, True)}

if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 1, 2]
    prim = pd.read_csv(f'{D}/data/processed/primary_2009.csv')
    ext = pd.read_csv(f'{D}/data/processed/external_rokomari_clean.csv')
    T = [clean(t) for t in prim.text]
    Y = prim.label.values
    ET = [clean(t) for t in ext.text]
    EY = ext.label.values
    rows, t0 = [], time.time()
    probs = {}  # name -> (oof prob matrix averaged over seeds)
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(T, Y)):
            tr_t, tr_y = [T[i] for i in tr], Y[tr].tolist()
            te_t = [T[i] for i in te]
            for name, (c, bi, at) in CONFIGS.items():
                m, v = fit(c, bi, at, tr_t, tr_y, seed * 100 + fold)
                pt, pe = predict(m, v, te_t), predict(m, v, ET)
                probs.setdefault(name, {})[(seed, fold)] = (te, pt)
                r = dict(model=name, seed=seed, fold=fold, **{f'cv_{k}': x for k, x in metrics(Y[te], pt.argmax(1)).items()},
                         **{f'ext_{k}': x for k, x in metrics(EY, pe.argmax(1)).items()})
                rows.append(r)
                print(f'{time.time()-t0:6.0f}s seed{seed} fold{fold} {name:12s} cv_acc={r["cv_acc"]:.3f} ext_acc={r["ext_acc"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(f'{OUT}/cv_rnn_raw.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/cv_rnn_raw.csv', index=False)
    print(df.groupby('model')[['cv_acc', 'cv_f1', 'ext_acc']].agg(['mean', 'std']).round(4).to_string())
