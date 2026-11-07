# BBR-2009: Bengali book review sentiment, cross-validated benchmark

See `HANDOFF.md` for the current status, open tasks and how to rebuild everything. Single branch: `master`.

Code, data release and result tables for the paper *Beyond the Single Split: A Reproducible Benchmark of Bengali Book Review Sentiment Classification with Explanations and Label-Noise Analysis* (`paper/paper.pdf`).

## What is here
- `data/BBR-2009.csv` : the 2,009-review corpus (980 positive, 1,029 negative) with the exact 5-fold assignments for seeds 0-2. See `data/DATASHEET.md` and `data/LICENSE-DATA.md`. **Non-commercial use only (CC BY-NC-SA 4.0).**
- `src/` : all experiment code (CV of recurrent, classical and transformer models, SHAP, McNemar/ECE, transfer, BanglaBook, figures).
- `results/` : result tables (CSV/JSON) behind every number in the paper. Raw out-of-fold probability arrays are not included (size).
- `figures/` : the paper figures (PDF).
- `paper/` : `paper.pdf` (compiled, A4, 6 pages), `bbr2009_overleaf.zip` (upload it to Overleaf as a project and compile with pdfLaTeX) and the same project unzipped in `paper/overleaf/`.

## Reproducing
Run everything from the repository root (paths in the scripts are relative to it).
1. `pip install -r requirements.txt`
2. Download the public sources yourself (we do not redistribute them beyond BBR-2009): Mendeley Bengali Book Reviews (doi 10.17632/2wcw3sxxr3.1) to `data/mendeley/BookReviews.xlsx`, the Rokomari reviews (github.com/gcsarker/Bangla-Book-Review-Dataset) to `data/rokomari/annotated_reviews.xlsx`, and BanglaBook (`train/validation/test.csv`) to `data/banglabook/`.
3. Download the models (`csebuetnlp/banglabert`, `csebuetnlp/banglabert_large`, `FacebookAI/xlm-roberta-base`, `google-bert/bert-base-multilingual-cased`) into `models/<name>/`.
4. `python src/build_primary.py` rebuilds `data/processed/primary_2009.csv` (the corpus in `data/BBR-2009.csv`).
5. Experiments: `src/cv_rnn.py`, `src/cv_classical.py <dataset> <seed>`, `BB_MODEL=banglabert BB_TAG= python src/cv_bert.py primary 0,1,2 4` (see the docstring of each script for options), `src/explain2.py`, `src/analysis2.py`, `src/transfer.py`, `src/rokomari_protocol.py`, `src/bert_banglabook_v4.py` and `src/ensemble_bb.py`.
A GPU is needed for the transformers (we used one RTX-class GPU with bf16).

## Honest notes
- BBR-2009 is a reconstruction from public sources; the BanglaBook part is almost entirely negative, so source and label are confounded.
- On the official BanglaBook benchmark our result (0.927 weighted F1) is below the published 0.933.
- The label-noise audit used one LLM annotator, not humans.

## Citation
To be added after review.
