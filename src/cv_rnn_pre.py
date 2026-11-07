"""Recurrent baselines with PRETRAINED (frozen) Bengali subword embeddings taken from BanglaBERT's input embedding matrix.
Same protocol as cv_rnn.py (5-fold, inner 10% early stopping). usage: cv_rnn_pre.py <seeds>"""
import sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModel, AutoTokenizer
import cv_rnn as c
D = '.'; dev = torch.device('cpu'); c.dev = dev
tok = AutoTokenizer.from_pretrained(f'{D}/models/banglabert')
W = AutoModel.from_pretrained(f'{D}/models/banglabert').get_input_embeddings().weight.detach().clone()
V, E = W.shape; print('pretrained embeddings', tuple(W.shape), flush=True)


class PreVocab:
    def batch(self, texts):
        seqs = [tok(t, add_special_tokens=False, truncation=True, max_length=c.MAXLEN)['input_ids'] or [1] for t in texts]
        L = torch.tensor([len(s) for s in seqs]); X = torch.zeros(len(seqs), max(L), dtype=torch.long)
        for i, s in enumerate(seqs): X[i, :len(s)] = torch.tensor(s)
        return X, L


class PreNet(c.Net):
    def __init__(self, cell, bi, attn):
        super().__init__(cell, bi, attn)
        self.emb = nn.Embedding.from_pretrained(W.clone(), freeze=True, padding_idx=0)
        d = c.HID * (2 if bi else 1)
        self.rnn = {'gru': nn.GRU, 'lstm': nn.LSTM}[cell](E, c.HID, batch_first=True, bidirectional=bi)


def fit(cell, bi, attn, tr_t, tr_y, seed):
    torch.manual_seed(seed); a_t, v_t, a_y, v_y = train_test_split(tr_t, tr_y, test_size=0.1, stratify=tr_y, random_state=seed)
    vocab = PreVocab(); model = PreNet(cell, bi, attn).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    cw = torch.tensor(len(a_y) / (2 * np.bincount(a_y)), dtype=torch.float, device=dev); lossf = nn.CrossEntropyLoss(weight=cw)
    best, st, bad = 1e9, None, 0; rng = np.random.RandomState(seed)
    for ep in range(c.EPOCHS):
        model.train(); idx = rng.permutation(len(a_t))
        for i in range(0, len(idx), c.BATCH):
            b = idx[i:i + c.BATCH]; X, L = vocab.batch([a_t[j] for j in b]); y = torch.tensor([a_y[j] for j in b], device=dev)
            opt.zero_grad(); lossf(model(X.to(dev), L), y).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        p = c.predict(model, vocab, v_t); vl = float(-np.log(p[np.arange(len(v_y)), v_y] + 1e-9).mean())
        if vl < best - 1e-4: best, bad = vl, 0; st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c.PATIENCE: break
    model.load_state_dict(st); return model, vocab


if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1].split(',')]
    prim = pd.read_csv(f'{D}/data/processed/primary_2009.csv').drop_duplicates('text').reset_index(drop=True)
    T = [c.clean(t) for t in prim.text]; Y = prim.label.values; rows = []; t0 = time.time()
    CF = {'BiLSTM+Attn': ('lstm', True, True), 'BiGRU+Attn': ('gru', True, True), 'BiLSTM': ('lstm', True, False)}
    for seed in seeds:
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(T, Y)):
            for name, (cell, bi, at) in CF.items():
                m, v = fit(cell, bi, at, [T[i] for i in tr], Y[tr].tolist(), seed * 100 + fold)
                p = c.predict(m, v, [T[i] for i in te]).argmax(1); r = dict(model=name, seed=seed, fold=fold, acc=float((p == Y[te]).mean()))
                rows.append(r); print(f'{time.time()-t0:5.0f}s s{seed} f{fold} {name:12s} {r["acc"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(f'{D}/results/cv_rnn_pre.csv', index=False)
    df = pd.DataFrame(rows); print(df.groupby('model').acc.agg(['mean', 'std']).round(4))
