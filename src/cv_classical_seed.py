"""Classical ML baselines on TF-IDF (word 1-2gram + char 2-5gram), same stratified 5-fold CV as the neural runs.
usage: cv_classical.py <primary|mendeley|rokomari> [seed]"""
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
D = '.'
ds = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def clean(t):
    return re.sub(r'\s+', ' ', str(t)).strip()


if ds == 'primary':
    d = pd.read_csv(f'{D}/data/processed/primary_2009.csv')
elif ds == 'rokomari':
    d = pd.read_excel(f'{D}/data/rokomari/annotated_reviews.xlsx')
    d = d[d.category.isin(['positive', 'negative'])].rename(columns={'review': 'text'})
    d['label'] = (d.category == 'positive').astype(int)
else:
    d = pd.read_excel(f'{D}/data/mendeley/BookReviews.xlsx').rename(columns={'Review': 'text', 'Sentiment': 'label'})
d = d.drop_duplicates('text')
T, Y = [clean(t) for t in d.text], d.label.values.astype(int)

# (name, factory, representation) ; 'sparse' = TF-IDF matrix, 'dense' = TF-IDF -> SVD(200)
MODELS = [
    ('LogisticRegression', lambda: LogisticRegression(max_iter=2000, C=10, class_weight='balanced'), 'sparse'),
    ('LinearSVM', lambda: LinearSVC(C=1, class_weight='balanced'), 'sparse'),
    ('RBF-SVM', lambda: SVC(C=10, kernel='rbf', class_weight='balanced'), 'sparse'),
    ('KNN', lambda: KNeighborsClassifier(n_neighbors=7, metric='cosine'), 'sparse'),
    ('DecisionTree', lambda: DecisionTreeClassifier(random_state=seed), 'sparse'),
    ('RandomForest', lambda: RandomForestClassifier(300, n_jobs=8, random_state=seed), 'sparse'),
    ('ExtraTrees', lambda: ExtraTreesClassifier(300, n_jobs=8, random_state=seed), 'sparse'),
    ('MultinomialNB', lambda: MultinomialNB(alpha=0.1), 'sparse'),
    ('BernoulliNB', lambda: BernoulliNB(alpha=0.1), 'sparse'),
    ('GaussianNB', lambda: GaussianNB(), 'dense'),
    ('LDA', lambda: LinearDiscriminantAnalysis(), 'dense'),
    ('QDA', lambda: QuadraticDiscriminantAnalysis(reg_param=0.5), 'dense'),
    ('XGBoost', lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=8, random_state=seed, verbosity=0), 'sparse'),
    ('LightGBM', lambda: LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=8, random_state=seed, verbose=-1), 'sparse'),
    ('CatBoost', lambda: CatBoostClassifier(iterations=300, depth=6, verbose=0, thread_count=8, random_seed=seed), 'dense'),
    ('GradientBoosting', lambda: GradientBoostingClassifier(random_state=seed), 'dense'),
    ('HistGradientBoosting', lambda: HistGradientBoostingClassifier(random_state=seed), 'dense'),
    ('AdaBoost', lambda: AdaBoostClassifier(random_state=seed), 'dense'),
    ('MLP', lambda: MLPClassifier((256,), early_stopping=True, max_iter=300, random_state=seed), 'sparse'),
]


def feats(tr, te):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r'(?u)[ঀ-৿\w]+')
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000)
    Xtr, Xte = hstack([w.fit_transform(tr), c.fit_transform(tr)]).tocsr(), hstack([w.transform(te), c.transform(te)]).tocsr()
    svd = TruncatedSVD(200, random_state=seed)
    return Xtr, Xte, svd.fit_transform(Xtr), svd.transform(Xte)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1)


rows, t0 = [], time.time()
for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(T, Y)):
    Xtr, Xte, Str, Ste = feats([T[i] for i in tr], [T[i] for i in te])
    for name, mk, rep in MODELS:
        a, b = (Xtr, Xte) if rep == 'sparse' else (Str, Ste)
        try:
            m = mk().fit(a, Y[tr])
            r = dict(dataset=ds, model=name, seed=seed, fold=fold, **metrics(Y[te], m.predict(b)))
        except Exception as e:
            r = dict(dataset=ds, model=name, seed=seed, fold=fold, acc=np.nan, prec=np.nan, rec=np.nan, f1=np.nan, err=str(e)[:80])
        rows.append(r)
        print(f'{time.time()-t0:5.0f}s fold{fold} {name:20s} acc={r["acc"]:.4f}', flush=True)
    pd.DataFrame(rows).to_csv(f'{D}/results/cv_classical_{ds}_s{seed}.csv', index=False)
df = pd.DataFrame(rows)
print(df.groupby('model')[['acc', 'f1']].agg(['mean', 'std']).round(4).sort_values(('acc', 'mean'), ascending=False).to_string())
