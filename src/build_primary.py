"""Rebuild the expanded primary dataset (target: 1,029 neg / 980 pos) and a leakage-free external set."""
import re
import pandas as pd

SEED = 42
D = './data'
BN = re.compile(r'[ঀ-৿]')


def norm(t):
    t = re.sub(r'[^ঀ-৿\s]', ' ', str(t))
    return re.sub(r'\s+', ' ', t).strip()


def bangla_ok(t, min_words=3):
    t = str(t)
    letters = re.sub(r'[\s\d\W]', '', t)
    return len(letters) > 0 and len(BN.findall(t)) / len(letters) > 0.9 and len(norm(t).split()) >= min_words


men = pd.read_excel(f'{D}/mendeley/BookReviews.xlsx')[['Review', 'Sentiment']]
men.columns = ['text', 'label']
men['source'] = 'mendeley'
men['key'] = men.text.map(norm)
men = men[men.key.str.len() > 0].drop_duplicates('key').drop(columns='key').reset_index(drop=True)
seen = set(men.text.map(norm))

bb = pd.concat([pd.read_csv(f'{D}/banglabook/{s}.csv') for s in ['train', 'validation', 'test']])
bb = bb[bb.Rating.isin([1, 2, 5])]
bb['label'] = (bb.Rating == 5).astype(int)
bb = bb.rename(columns={'Review': 'text'})
bb = bb[bb.text.map(bangla_ok)]
bb['key'] = bb.text.map(norm)
bb = bb[~bb.key.isin(seen)].drop_duplicates('key')

roko = pd.read_excel(f'{D}/rokomari/annotated_reviews.xlsx')
roko_keys = set(roko.review.map(norm))

need = {0: 1029 - int((men.label == 0).sum()), 1: 980 - int((men.label == 1).sum())}
print('need extra', need)
parts = []
for lab, n in need.items():
    pool = bb[bb.label == lab]
    wafi = pool[pool.Site == 'WafiLife']
    clean = pool[(pool.Site != 'WafiLife') & ~pool.key.isin(roko_keys)]
    take = wafi.sample(min(n, len(wafi)), random_state=SEED)
    if len(take) < n:
        take = pd.concat([take, clean.sample(n - len(take), random_state=SEED)])
    print('label', lab, 'wafilife', int((take.Site == 'WafiLife').sum()), 'other', int((take.Site != 'WafiLife').sum()))
    parts.append(take)
ext = pd.concat(parts)[['text', 'label']]
ext['source'] = 'banglabook'
prim = pd.concat([men, ext], ignore_index=True)
prim['key'] = prim.text.map(norm)
prim = prim.drop_duplicates('key').drop(columns='key').reset_index(drop=True)
prim.to_csv(f'{D}/processed/primary_2009.csv', index=False)
print(prim.shape, prim.label.value_counts().to_dict(), prim.source.value_counts().to_dict())

# external: Rokomari, drop neutral (rating 3), rating<=2 neg / >=4 pos, remove anything seen in primary
pk = set(prim.text.map(norm))
roko = roko[roko.rating != 3].copy()
roko['label'] = (roko.rating >= 4).astype(int)
roko['key'] = roko.review.map(norm)
roko = roko[roko.review.map(bangla_ok) & ~roko.key.isin(pk)].drop_duplicates('key')
roko = roko.rename(columns={'review': 'text'})[['text', 'label', 'rating']]
roko.to_csv(f'{D}/processed/external_rokomari_clean.csv', index=False)
print('external', roko.shape, roko.label.value_counts().to_dict())
