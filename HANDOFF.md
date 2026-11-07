# Handoff: where the project stands and how to continue from GitHub alone

Branch policy: everything lives on the single branch `master`.

## What the project is
Cross-validated benchmark of Bengali book-review sentiment classification (recurrent, TF-IDF, BanglaBERT, XLM-R, mBERT), with SHAP explanations, a label-noise analysis, and a comparison with the official BanglaBook and Rokomari results. Corpus built here: **BBR-2009** (1,414 Mendeley + 595 BanglaBook reviews).

## Deliverables (all in this repository)
| What | Where |
|---|---|
| Conference paper PDF (IEEE format, A4, 6 pages, anonymous) | `paper/paper.pdf` |
| Overleaf project (upload the zip, compile with pdfLaTeX) | `paper/bbr2009_overleaf.zip`, unzipped in `paper/overleaf/` |
| LaTeX sources | `latex/conference/` (6-page paper), `latex/journal/` (longer version) |
| Word draft (anonymous, authors removed) | `docx/paper-word-draft-anonymous.docx` |
| Data release (corpus, folds, datasheet) | `data/BBR-2009.csv`, `data/DATASHEET.md` |
| All code | `src/` (scripts hold relative paths: run from the repository root) |
| Every result table and the saved predictions | `results/` |
| Number audit (re-computes every number in the paper) | `src/audit_paper.py` (125 checks, 0 mismatches at the last run) |
| Human annotation packet (blind) | `annotation/annotator1.csv`, `annotation/GUIDELINE.md` |

## Main results (all verified by the audit)
- BBR-2009, 5-fold CV: BanglaBERT 93.84 +- 1.10 (3 seeds), BanglaBERT-large 93.81 +- 1.04, ensemble 94.23 +- 0.82 (not significant), XLM-R 91.31, mBERT 86.83, TF-IDF Linear SVM 90.19, recurrent from scratch at most 81.08, recurrent with frozen pretrained embeddings 87.61.
- Source shortcut: predicting the label from the source alone scores 75.1%.
- Mendeley: BanglaBERT 96.72 +- 1.19, TF-IDF Naive Bayes 94.24. Rokomari (protocol of Sarker et al., 3 shuffles): 89.97 +- 0.66 weighted F1.
- Official BanglaBook (3-class): ensemble 0.927 weighted F1, **0.6 below the published 0.9331** (verified in Table 3 of the ACL paper). Six runs, all 0.923-0.926 on test.
- Label noise: 69 of 111 errors are confident (p>0.9); 8.6% of BanglaBook reviews against 1.3% of Mendeley reviews. A blind audit by a single language-model annotator agrees with the dataset label on 96.6% of random reviews but only 25.9% of the suspected ones.
- Honest framing: no state-of-the-art claim.

## Still to do (cannot be done by code alone)
1. Human annotation of `annotation/annotator1.csv` (200 reviews, label column `label_P_N_U`); do not open `results/annot_*.csv` before finishing. Then compute Cohen's kappa and update the label-noise section.
2. Choose the venue and move the paper to its template and page limit.
3. The paper currently links the private GitHub repository. For a double-blind review, replace that link (Availability section) with an anonymous mirror URL (for example anonymous.4open.science), because the GitHub link shows the account name; for single-blind or camera-ready, make the repository public first.
4. Check two reference page ranges at the source (Guo et al., ICML 2017, pp. 1321-1330; Vaswani et al., NeurIPS 2017, pp. 5998-6008).
5. Supervisor review and proofreading; fill in real author names and affiliations only for the final version.
6. Optional extras that would strengthen the paper: few-shot large language model baseline, a second human annotator, tuned recurrent/classical baselines.

## How to rebuild everything
1. `pip install -r requirements.txt`
2. Rebuild the BanglaBook split: see `data/banglabook_parts/README.md`.
3. Download the models into `models/` (not stored here, 3.7 GB): `csebuetnlp/banglabert`, `csebuetnlp/banglabert_large`, `FacebookAI/xlm-roberta-base`, `google-bert/bert-base-multilingual-cased`.
4. Re-run the numbers: `python src/audit_paper.py` (needs only `results/` and `data/`; no GPU).
5. Re-run experiments with the scripts in `src/` (see `README.md`). A GPU is needed for the transformers.
6. Paper: upload `paper/bbr2009_overleaf.zip` to Overleaf, or `cd latex/conference && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
7. Figures: `python src/figs_conf.py`.

## Things to remember
- The corpus is a reconstruction from public sources. The BanglaBook part is 94.5% negative, so source and label are confounded (see the caution paragraph in the paper).
- The 84% Mendeley baseline of Hossain et al. comes from their own 2,000-review corpus, of which only 1,443 reviews are public.
- Earlier BanglaBook runs were adjusted after seeing test scores; the paper says so. The final ensemble was chosen on validation only.
- Licence: code MIT; data CC BY-NC-SA 4.0 (inherited from BanglaBook), non-commercial use only.
- The original Word file with real author names is deliberately not stored here.
