"""BanglaBERT-large on the official BanglaBook split (3-class: 0 neg, 1 neutral, 2 pos); reports weighted/macro F1 on test."""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

D = '.'
MP = f'{D}/models/banglabert_large'
dev = torch.device('cuda')
MAXLEN, BS, LR, EPOCHS = 128, 16, 2e-5, 3
torch.manual_seed(0)
rd = lambda s: pd.read_csv(f'{D}/data/banglabook/{s}.csv')
tr, va, te = rd('train'), rd('validation'), rd('test')
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForSequenceClassification.from_pretrained(MP, num_labels=3).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
Ttr, Ytr = tr.Review.astype(str).tolist(), tr.label.values
CW = torch.tensor((len(Ytr) / (3 * np.bincount(Ytr))) ** 0.5, dtype=torch.float, device=dev)
steps = EPOCHS * ((len(Ttr) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)


def predict(texts, bs=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            b = tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
            out[idx] = torch.softmax(model(**b).logits.float(), 1).cpu().numpy()
    return out


def report(df, name):
    p = predict(df.Review.astype(str).tolist())
    y = df.label.values
    r = dict(split=name, acc=accuracy_score(y, p.argmax(1)), weighted_f1=f1_score(y, p.argmax(1), average='weighted'),
             macro_f1=f1_score(y, p.argmax(1), average='macro'))
    return r, p


t0, rng, best = time.time(), np.random.RandomState(0), -1
for ep in range(EPOCHS):
    model.train()
    idx = rng.permutation(len(Ttr))
    for k, i in enumerate(range(0, len(idx), BS)):
        b = idx[i:i + BS]
        enc = tok([Ttr[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**enc).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(Ytr[b], device=dev), weight=CW)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if k % 1000 == 0:
            print(f'{time.time()-t0:6.0f}s epoch {ep} step {k} loss {loss.item():.4f}', flush=True)
    rv, _ = report(va, 'validation')
    print(rv, flush=True)
    if rv['weighted_f1'] > best:
        best = rv['weighted_f1']
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
rt, pt = report(te, 'test')
print(rt, flush=True)
np.save(f'{D}/results/banglabook_v2_test_prob.npy', pt)
json.dump(dict(validation_best_weighted_f1=best, test=rt, epochs=EPOCHS, lr=LR, bs=BS, maxlen=MAXLEN), open(f'{D}/results/banglabook_v2_result.json', 'w'), indent=1)
print('DONE')
