"""BanglaBook 3-class, robust fine-tuning. Env: BK_MODEL, BK_LR, BK_EPOCHS, BK_BS, BK_CWPOW (class weight exponent), BK_SEED, BK_TAG.
Evaluates on validation twice per epoch, keeps the best-by-validation weighted-F1 checkpoint, and saves validation + test probabilities.
Model selection uses validation only."""
import json, os, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
D = '.'
E = os.environ.get
MP = f"{D}/models/{E('BK_MODEL', 'banglabert_large')}"; LR = float(E('BK_LR', 1e-5)); EPOCHS = int(E('BK_EPOCHS', 3)); BS = int(E('BK_BS', 16))
CWPOW = float(E('BK_CWPOW', 0)); SEED = int(E('BK_SEED', 1)); TAG = E('BK_TAG', 'v4a'); MAXLEN = 128
dev = torch.device('cuda'); torch.manual_seed(SEED)
rd = lambda s: pd.read_csv(f'{D}/data/banglabook/{s}.csv')
tr, va, te = rd('train'), rd('validation'), rd('test')
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForSequenceClassification.from_pretrained(MP, num_labels=3).to(dev)
T, Y = tr.Review.astype(str).tolist(), tr.label.values
cw = torch.tensor((len(Y) / (3 * np.bincount(Y))) ** CWPOW, dtype=torch.float, device=dev)
nd = ['bias', 'LayerNorm.weight']
opt = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if not any(x in n for x in nd)], 'weight_decay': 0.01},
                         {'params': [p for n, p in model.named_parameters() if any(x in n for x in nd)], 'weight_decay': 0.0}], lr=LR)
spe = (len(T) + BS - 1) // BS; steps = EPOCHS * spe
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)


def predict(texts, bs=256):
    model.eval(); order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 3), np.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            b = tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
            out[idx] = torch.softmax(model(**b).logits.float(), 1).cpu().numpy()
    model.train(); return out


def score(p, y):
    return dict(acc=accuracy_score(y, p.argmax(1)), wf1=f1_score(y, p.argmax(1), average='weighted'), mf1=f1_score(y, p.argmax(1), average='macro'))


vt, vy, tt, ty = va.Review.astype(str).tolist(), va.label.values, te.Review.astype(str).tolist(), te.label.values
rng = np.random.RandomState(SEED); best = -1; t0 = time.time(); step = 0; log = []
model.train()
for ep in range(EPOCHS):
    idx = rng.permutation(len(T))
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        enc = tok([T[j] for j in b], truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt').to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(**enc).logits
        loss = torch.nn.functional.cross_entropy(lg.float(), torch.tensor(Y[b], device=dev), weight=cw if CWPOW > 0 else None)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); step += 1
        if step % (spe // 2) == 0:
            pv = predict(vt); s = score(pv, vy); log.append(dict(step=step, **s)); print(f'{time.time()-t0:6.0f}s ep{ep} step{step} val {s}', flush=True)
            if s['wf1'] > best:
                best = s['wf1']; pt = predict(tt); np.save(f'{D}/results/bb_{TAG}_val.npy', pv); np.save(f'{D}/results/bb_{TAG}_test.npy', pt)
                json.dump(dict(val=s, test=score(pt, ty), log=log, env=dict(model=MP, lr=LR, epochs=EPOCHS, bs=BS, cwpow=CWPOW, seed=SEED)), open(f'{D}/results/bb_{TAG}.json', 'w'), indent=1)
print('DONE best val wf1', best, flush=True)
