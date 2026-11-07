"""Build a public, reproducible release of BBR-2009: ids, text, label, source, and the exact 5-fold splits (seeds 0-2).
Text is redistributed under the original licences (Mendeley CC BY 4.0; BanglaBook CC BY-NC-SA 4.0 -> release is non-commercial)."""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
D = '.'
out = f'{D}/release'; os.makedirs(out, exist_ok=True)
d = pd.read_csv(f'{D}/data/processed/primary_2009.csv').drop_duplicates('text').reset_index(drop=True)
d.insert(0, 'id', [f'BBR{i:04d}' for i in range(len(d))])
for s in range(3):
    d[f'fold_seed{s}'] = -1
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(d.text, d.label)):
        d.loc[te, f'fold_seed{s}'] = k
sus = pd.read_csv(f'{D}/results/suspect_labels.csv')
d['suspected_noisy'] = d.text.isin(set(sus.text)).astype(int)
d.to_csv(f'{out}/BBR-2009.csv', index=False)
open(f'{out}/DATASHEET.md', 'w').write(f"""# BBR-2009 datasheet
Binary (0 negative / 1 positive) Bengali book-review sentiment, {len(d)} unique reviews: {int((d.label==0).sum())} negative, {int((d.label==1).sum())} positive.
Sources: Mendeley Bengali Book Reviews (doi 10.17632/2wcw3sxxr3.1, CC BY 4.0; manual labels) and BanglaBook (Kabir et al., ACL Findings 2023, CC BY-NC-SA 4.0; labels derived from star ratings).
Per source: {d.source.value_counts().to_dict()}.
Columns: id, text, label, source, fold_seed0..2 (stratified 5-fold assignment used in the paper), suspected_noisy (model-flagged, NOT manually verified).
Known issues: BanglaBook rows are ~95% negative, so source and label are confounded; rating-derived labels are noisy.
Licence of this release: CC BY-NC-SA 4.0 (inherits the most restrictive source licence). Non-commercial use only.
""")
print(d.groupby('source').label.agg(['count', 'mean']), d.suspected_noisy.sum())
