"""BanglaBERT 5-fold CV fine-tuning. usage: cv_bert.py <dataset: primary|mendeley> <seeds comma list> [epochs]"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

D = '.'
MP = f'{D}/models/' + os.environ.get('BB_MODEL', 'banglabert')
TAG = os.environ.get('BB_TAG', '')
USE_CW = os.environ.get('BB_CW', '1') == '1'
dev = torch.device('cuda')
MAXLEN, BS, LR = int(os.environ.get('BB_MAXLEN', 128)), int(os.environ.get('BB_BS', 16)), float(os.environ.get('BB_LR', 3e-5))


def clean(t):
    return re.sub(r'\s+', ' ', str(t)).strip()


def load(name):
    if name == 'primary':
        d = pd.read_csv(f'{D}/data/processed/primary_2009.csv')
    elif name == 'rokomari':
        d = pd.read_excel(f'{D}/data/rokomari/annotated_reviews.xlsx')
        d = d[d.category.isin(['positive', 'negative'])].rename(columns={'review': 'text'})
        d['label'] = (d.category == 'positive').astype(int)
    else:
        d = pd.read_excel(f'{D}/data/mendeley/BookReviews.xlsx').rename(columns={'Review': 'text', 'Sentiment': 'label'})
    d = d.drop_duplicates('text')
    return [clean(t) for t in d.text], d.label.values.astype(int)


def probs(model, tok, texts, bs=64):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
            out.append(torch.softmax(model(**b).logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    w = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)[2]
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1, wf1=w)


def train(tr_t, tr_y, seed, epochs):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MP)
    model = AutoModelForSequenceClassification.from_pretrained(MP, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * ((len(tr_t) + BS - 1) // BS)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    cw = torch.tensor(len(tr_y) / (2 * np.bincount(tr_y)), dtype=torch.float, device=dev) if USE_CW else None
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(tr_t))
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            enc = tok([tr_t[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
            y = torch.tensor(tr_y[b], device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = lossf(logits.float(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
    return model, tok


if __name__ == '__main__':
    ds = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(',')]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    T, Y = load(ds)
    ext = pd.read_csv(f'{D}/data/processed/external_rokomari_clean.csv')
    ET, EY = [clean(t) for t in ext.text], ext.label.values
    rows, t0 = [], time.time()
    for seed in seeds:
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(T, Y)):
            m, tok = train([T[i] for i in tr], Y[tr], seed * 100 + fold, epochs)
            pt = probs(m, tok, [T[i] for i in te])
            pe = probs(m, tok, ET) if ds != 'rokomari' else np.zeros((len(ET), 2))  # external overlaps rokomari
            np.save(f'{D}/results/bert{TAG}_{ds}_s{seed}_f{fold}_idx.npy', te)
            np.save(f'{D}/results/bert{TAG}_{ds}_s{seed}_f{fold}_prob.npy', pt)
            r = dict(dataset=ds, seed=seed, fold=fold, **{f'cv_{k}': v for k, v in metrics(Y[te], pt.argmax(1)).items()},
                     **{f'ext_{k}': v for k, v in metrics(EY, pe.argmax(1)).items()})
            rows.append(r)
            print(f'{time.time()-t0:6.0f}s {ds} seed{seed} fold{fold} cv_acc={r["cv_acc"]:.4f} cv_f1={r["cv_f1"]:.4f} ext_acc={r["ext_acc"]:.4f}', flush=True)
            pd.DataFrame(rows).to_csv(f'{D}/results/cv_bert{TAG}_{ds}.csv', index=False)
            del m
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df[['cv_acc', 'cv_f1', 'cv_wf1', 'ext_acc']].agg(['mean', 'std']).round(4).to_string())
