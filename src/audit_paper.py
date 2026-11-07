"""Recompute every number quoted in the conference paper from raw results and compare with the stated value."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
R = 'results/'; bad = []; n_ok = 0
def chk(name, got, exp, tol=0.006):
    global n_ok
    ok = abs(got - exp) <= tol
    n_ok += ok
    if not ok: bad.append((name, round(float(got), 4), exp))
d = pd.read_csv('data/processed/primary_2009.csv').drop_duplicates('text').reset_index(drop=True)
Y = d.label.values; S = d.source.values
chk('n', len(d), 2009, 0); chk('pos', (Y == 1).sum(), 980, 0); chk('neg', (Y == 0).sum(), 1029, 0)
chk('mendeley n', (S == 'mendeley').sum(), 1414, 0); chk('bb n', (S == 'banglabook').sum(), 595, 0)
chk('bb neg', ((S == 'banglabook') & (Y == 0)).sum(), 562, 0); chk('bb pos', ((S == 'banglabook') & (Y == 1)).sum(), 33, 0)
# recurrent
r = pd.read_csv(R + 'cv_rnn_raw.csv'); g = r.groupby('model').cv_acc
chk('BiLSTM+Attn mean', g.mean()['BiLSTM+Attn'] * 100, 81.08, .01); chk('BiLSTM+Attn std', g.std()['BiLSTM+Attn'] * 100, 3.75, .01)
chk('RNN mean', g.mean()['RNN'] * 100, 73.57, .01); chk('RNN std', g.std()['RNN'] * 100, 6.20, .01)
chk('fold min BiLSTM+Attn', g.min()['BiLSTM+Attn'] * 100, 72.6, .06); chk('fold max BiLSTM+Attn', g.max()['BiLSTM+Attn'] * 100, 85.8, .06)
chk('std range lo', g.std().min() * 100, 3.58, .05); chk('std range hi', g.std().max() * 100, 6.72, .05)
# bert 3 seeds
def runs(tag):
    out = []
    for s in range(3):
        for f in range(5):
            i = np.load(f'{R}bert{tag}_primary_s{s}_f{f}_idx.npy'); p = np.load(f'{R}bert{tag}_primary_s{s}_f{f}_prob.npy')
            out.append((s, f, i, p))
    return out
A = {}
for t in ['', '_large']: A[t] = runs(t)
def accs(rs, ens=None):
    o = []
    for k, (s, f, i, p) in enumerate(rs):
        q = p if ens is None else (p + ens[k][3]) / 2
        o.append((q.argmax(1) == Y[i]).mean())
    return np.array(o)
ab, al = accs(A['']), accs(A['_large']); ae = accs(A[''], A['_large'])
for n, a, m, sd in [('base', ab, 93.84, 1.10), ('large', al, 93.81, 1.04), ('ens', ae, 94.23, 0.82)]:
    chk(n + ' mean', a.mean() * 100, m, .01); chk(n + ' std', a.std(ddof=1) * 100, sd, .02)
chk('wilcoxon large-base p', wilcoxon(al, ab).pvalue, 0.887, .01); chk('wilcoxon ens-base p', wilcoxon(ae, ab).pvalue, 0.162, .01)
for t, m, sd in [('_xlmr_tuned', 91.31, 1.12), ('_mbert_tuned', 86.83, 1.54)]:
    x = pd.read_csv(f'{R}cv_bert{t}_primary.csv').cv_acc; chk(t + ' mean', x.mean() * 100, m, .01); chk(t + ' std', x.std() * 100, sd, .02)
for t, m, sd in [('_abl3_default', 93.84, 1.10), ('_abl3_lr5e5', 93.86, 1.31), ('_abl3_nocw', 93.83, 1.27), ('_abl3_len64', 93.26, 1.42), ('_abl3_ep2', 93.13, 1.21)]:
    x = pd.read_csv(f'{R}cv_bert{t}_primary.csv').cv_acc; chk(t + ' mean', x.mean() * 100, m, .01); chk(t + ' std', x.std() * 100, sd, .02)
# classical
c = pd.concat([pd.read_csv(R + 'cv_classical_primary.csv').query('seed==0'), pd.read_csv(R + 'cv_classical_primary_s1.csv'), pd.read_csv(R + 'cv_classical_primary_s2.csv')])
cm = c.groupby('model').acc; chk('SVM mean', cm.mean()['LinearSVM'] * 100, 90.19, .01); chk('SVM std', cm.std()['LinearSVM'] * 100, 1.53, .02)
for ds, mdl, m in [('mendeley', 'MultinomialNB', 94.24), ('rokomari', 'LinearSVM', 88.42)]:
    x = pd.read_csv(f'{R}cv_classical_{ds}.csv').groupby('model').acc.mean() * 100; chk(ds + ' ' + mdl, x[mdl], m, .01)
x = pd.read_csv(R + 'cv_bert_mendeley.csv').cv_acc; chk('mendeley bert', x.mean() * 100, 96.72, .01); chk('mendeley bert sd', x.std() * 100, 1.19, .02)
x = pd.read_csv(R + 'cv_bert_rokomari.csv').cv_acc; chk('rokomari bert', x.mean() * 100, 90.60, .01); chk('rokomari bert sd', x.std() * 100, 0.72, .02)
# label noise
P = np.zeros((len(Y), 2)); cnt = np.zeros(len(Y))
for t in A:
    for s, f, i, p in A[t]: P[i] += p; cnt[i] += 1
P /= cnt[:, None]; pred = P.argmax(1); err = pred != Y; conf = P.max(1)
chk('errors', err.sum(), 111, 0); chk('confident errors', (err & (conf > .9)).sum(), 69, 0)
chk('conf pct', (err & (conf > .9)).sum() / err.sum() * 100, 62, .6)
sus = d.text.astype(str).str.strip().isin(set(pd.read_csv(R + 'suspect_labels.csv').text.astype(str).str.strip()))
chk('suspect bb', (sus & (S == 'banglabook')).sum(), 51, 0); chk('suspect mend', (sus & (S == 'mendeley')).sum(), 18, 0)
chk('err rate bb', err[S == 'banglabook'].mean() * 100, 12.1, .06); chk('err rate mend', err[S == 'mendeley'].mean() * 100, 2.8, .06)
d['len'] = d.text.astype(str).str.split().str.len(); neg = d.text.astype(str).str.contains(r'(^|\s)(না|নি|নেই|নয়|নাই)(\s|$|[।,.!])', regex=True)
chk('err neg word', err[neg].mean() * 100, 4.7, .06); chk('err no neg word', err[~neg].mean() * 100, 5.9, .06)
q = pd.qcut(d['len'], 4, duplicates='drop'); e = pd.Series(err).groupby(q.values, observed=True).mean() * 100
chk('len q min', e.min(), 4.2, .06); chk('len q max', e.max(), 7.0, .06)
# audit
a = pd.read_csv(R + 'annot_llm.csv'); a['given'] = a.label.map({1: 'P', 0: 'N'}); l = a[a.llm != 'U']
rd, sp = l[l.grp == 'random'], l[l.grp == 'suspect']
chk('random n', len(rd), 118, 0); chk('random agree', (rd.llm == rd.given).mean() * 100, 96.6, .06); chk('random kappa', cohen_kappa_score(rd.given, rd.llm), 0.93, .006)
chk('suspect n', len(sp), 54, 0); chk('suspect agree', (sp.llm == sp.given).mean() * 100, 25.9, .06); chk('suspect kappa', cohen_kappa_score(sp.given, sp.llm), 0.0, .01)
chk('unclear', (a.llm == 'U').sum(), 28, 0); chk('neg->pos', ((sp.given == 'N') & (sp.llm == 'P')).sum(), 39, 0); chk('given neg', (sp.given == 'N').sum(), 43, 0)
# explain + analysis + others
dt = json.load(open(R + 'explain2/deletion_test.json'))
for k, (sh, rn) in {'1': (.120, .024), '3': (.324, .095), '5': (.444, .150)}.items(): chk('del shap ' + k, dt[k]['shap_drop'], sh, .001); chk('del rnd ' + k, dt[k]['random_drop'], rn, .001)
an = json.load(open(R + 'analysis2.json')); chk('ece base', an['ece']['base'], .047, .001); chk('ece large', an['ece']['large'], .057, .001); chk('ece ens', an['ece']['ensemble'], .041, .001)
chk('mcnemar seed0 ens only', an['mcnemar_ens_vs_svm'][0]['ens_only_right'], 130, 0); chk('mcnemar seed0 svm only', an['mcnemar_ens_vs_svm'][0]['svm_only_right'], 48, 0)
chk('max mcnemar p vs svm <1e-8', max(x['p'] for x in an['mcnemar_ens_vs_svm']) < 1e-8, 1, 0)
for i, pv in enumerate([0.03, 0.87, 0.67]): chk('ens vs base p seed%d' % i, an['mcnemar_ens_vs_base'][i]['p'], pv, .01)
tr = {(x['train'], x['test']): x for x in json.load(open(R + 'transfer.json'))}
chk('M->R', tr[('mendeley', 'rokomari')]['acc'] * 100, 84.5, .06); chk('R->M', tr[('rokomari', 'mendeley')]['acc'] * 100, 91.2, .06)
chk('M->BB', tr[('mendeley', 'banglabook_subset')]['acc'] * 100, 77.0, .06); chk('R->BB', tr[('rokomari', 'banglabook_subset')]['acc'] * 100, 82.4, .06)
chk('M->BB f1', tr[('mendeley', 'banglabook_subset')]['macro_f1'] * 100, 58.3, .06); chk('R->BB f1', tr[('rokomari', 'banglabook_subset')]['macro_f1'] * 100, 63.8, .06)
rk = json.load(open(R + 'rokomari_protocol.json')); w = np.array([x['weighted_f1'] for x in rk]) * 100
chk('rok protocol mean', w.mean(), 89.97, .01); chk('rok protocol sd', w.std(ddof=1), 0.66, .01)
bb = json.load(open(R + 'bb_ensemble.json')); chk('BB ens wf1', bb['test']['wf1'], .927, .0006); chk('BB ens acc', bb['test']['acc'] * 100, 93.5, .06); chk('BB ens mf1', bb['test']['mf1'], .685, .001)
for t, v in [('v4a', .925), ('v4b', .926), ('v4c', .923)]:
    ty = pd.read_csv('data/banglabook/test.csv').label.values; chk('BB ' + t, f1_score(ty, np.load(f'{R}bb_{t}_test.npy').argmax(1), average='weighted'), v, .0006)
chk('BB v1', json.load(open(R + 'banglabook_result.json'))['test']['weighted_f1'], .892, .0006); chk('BB v2', json.load(open(R + 'banglabook_v2_result.json'))['test']['weighted_f1'], .9187, .0006)
chk('BB v3', json.load(open(R + 'banglabook_v3_result.json'))['test']['weighted_f1'], .846, .0006)
# per-class F1 of final ensemble
ty = pd.read_csv('data/banglabook/test.csv').label.values; vy = pd.read_csv('data/banglabook/validation.csv').label.values
pt = np.mean([np.load(f'{R}bb_{t}_test.npy') for t in bb['members']], 0); fin = np.log(pt + 1e-9) + np.array(list(bb['offsets']) + [0])
pc = f1_score(ty, fin.argmax(1), average=None); chk('per-class neg', pc[0], .71, .006); chk('per-class neu', pc[1], .37, .006); chk('per-class pos', pc[2], .97, .006)
# dataset table
x = pd.read_excel('data/rokomari/annotated_reviews.xlsx'); chk('rok pos', (x.category == 'positive').sum(), 3051, 0); chk('rok neg', (x.category == 'negative').sum(), 2585, 0)
m = pd.read_excel('data/mendeley/BookReviews.xlsx').rename(columns={'Review': 'text', 'Sentiment': 'label'}).drop_duplicates('text'); chk('mend n', len(m), 1442, 0); chk('mend pos', (m.label == 1).sum(), 970, 0)
e = pd.read_csv('data/processed/external_rokomari_clean.csv'); chk('ext n', len(e), 5277, 0); chk('ext pos', (e.label == 1).sum(), 2670, 0)
for sp_, n in [('train', 110645), ('validation', 15806), ('test', 31614)]: chk('bb ' + sp_, len(pd.read_csv(f'data/banglabook/{sp_}.csv')), n, 0)

# shortcut baselines and per-source accuracy
so = np.where(S == 'mendeley', 1, 0); chk('source-only acc', (so == Y).mean() * 100, 75.1, .06)
chk('bb majority', max(Y[S == 'banglabook'].mean(), 1 - Y[S == 'banglabook'].mean()) * 100, 94.5, .06)
chk('mend majority', Y[S == 'mendeley'].mean() * 100, 67.0, .06)
chk('ens acc bb', (pred[S == 'banglabook'] == Y[S == 'banglabook']).mean() * 100, 87.9, .06)
chk('ens acc mend', (pred[S == 'mendeley'] == Y[S == 'mendeley']).mean() * 100, 97.2, .06)

# pretrained-embedding recurrent baselines
pr = pd.read_csv(R + 'cv_rnn_pre.csv'); g2 = pr.groupby('model').acc
for mdl, m_, sd_ in [('BiLSTM', 87.61, 1.75), ('BiGRU+Attn', 87.37, 2.09), ('BiLSTM+Attn', 87.03, 2.12)]:
    chk('pre ' + mdl + ' mean', g2.mean()[mdl] * 100, m_, .01); chk('pre ' + mdl + ' sd', g2.std()[mdl] * 100, sd_, .02)
chk('pre runs', len(pr), 45, 0); chk('gap emb', g2.mean()['BiLSTM'] * 100 - 81.08, 6.5, .1)
print('checked OK:', n_ok, 'MISMATCHES:', len(bad))
for b in bad: print('  MISMATCH', b)
