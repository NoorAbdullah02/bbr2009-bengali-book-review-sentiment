"""McNemar (BanglaBERT ensemble vs LinearSVM, per seed), ECE/calibration, error analysis by group, from saved OOF predictions."""
import re, json
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
D = '.'; R = f'{D}/results'
d = pd.read_csv(f'{D}/data/processed/primary_2009.csv').drop_duplicates('text').reset_index(drop=True)
T = [re.sub(r'\s+', ' ', str(t)).strip() for t in d.text]; Y = d.label.values


def oof(tag, s):
    P = np.zeros((len(Y), 2))
    for f in range(5):
        i = np.load(f'{R}/bert{tag}_primary_s{s}_f{f}_idx.npy'); P[i] = np.load(f'{R}/bert{tag}_primary_s{s}_f{f}_prob.npy')
    return P


def svm(s):
    p = np.zeros(len(Y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(T, Y):
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r'(?u)[ঀ-৿\w]+')
        c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000)
        a = hstack([w.fit_transform([T[i] for i in tr]), c.fit_transform([T[i] for i in tr])]).tocsr()
        b = hstack([w.transform([T[i] for i in te]), c.transform([T[i] for i in te])]).tocsr()
        p[te] = LinearSVC(C=1, class_weight='balanced').fit(a, Y[tr]).predict(b)
    return p


def mcnemar(a, b):
    n01 = int(((a == Y) & (b != Y)).sum()); n10 = int(((a != Y) & (b == Y)).sum())
    st = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return n01, n10, float(1 - chi2.cdf(st, 1))


def ece(P, nb=10):
    conf, pred = P.max(1), P.argmax(1); acc = pred == Y; e = 0
    for lo in np.linspace(0, 1, nb, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


out = {'mcnemar_ens_vs_svm': [], 'mcnemar_ens_vs_base': [], 'ece': {}}
Pe_all = []
for s in range(3):
    pb, pl = oof('', s), oof('_large', s); pe = (pb + pl) / 2; Pe_all.append(pe)
    sv = svm(s)
    out['mcnemar_ens_vs_svm'].append(dict(seed=s, ens_only_right=mcnemar(pe.argmax(1), sv)[0], svm_only_right=mcnemar(pe.argmax(1), sv)[1], p=mcnemar(pe.argmax(1), sv)[2]))
    out['mcnemar_ens_vs_base'].append(dict(seed=s, p=mcnemar(pe.argmax(1), pb.argmax(1))[2]))
    for n, P in [('base', pb), ('large', pl), ('ensemble', pe)]:
        out['ece'].setdefault(n, []).append(ece(P))
out['ece'] = {k: float(np.mean(v)) for k, v in out['ece'].items()}
P = np.mean(Pe_all, 0); pred = P.argmax(1); err = pred != Y
d['err'] = err; d['len'] = [len(t.split()) for t in T]
d['neg_word'] = [bool(re.search(r'(^|\s)(না|নি|নেই|নয়|নাই)(\s|$|[।,.!])', t)) for t in T]
d['conf'] = P.max(1)
g = {}
g['by_source'] = d.groupby('source').err.agg(['mean', 'sum', 'count']).round(4).to_dict('index')
g['by_label'] = d.groupby('label').err.agg(['mean', 'sum', 'count']).round(4).to_dict('index')
g['by_length'] = d.groupby(pd.qcut(d.len, 4, duplicates='drop'), observed=True).err.agg(['mean', 'sum', 'count']).round(4).rename(index=str).to_dict('index')
g['by_negation_word'] = d.groupby('neg_word').err.agg(['mean', 'sum', 'count']).round(4).to_dict('index')
g['errors_conf_gt_0.9'] = int((d.err & (d.conf > 0.9)).sum()); g['errors_total'] = int(d.err.sum())
out['error_analysis'] = g
json.dump(out, open(f'{R}/analysis2.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
