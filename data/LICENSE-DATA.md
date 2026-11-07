# Data licence
`BBR-2009.csv` combines reviews from two public sources and is therefore released under **CC BY-NC-SA 4.0** (non-commercial, share-alike), the more restrictive of the two.

- Mendeley Bengali Book Reviews, doi 10.17632/2wcw3sxxr3.1 (CC BY 4.0). Cite the original dataset.
- BanglaBook, Kabir et al., Findings of ACL 2023, arXiv:2305.06595 (CC BY-NC-SA 4.0). Cite the original paper.

Columns: id, text, label (1 positive / 0 negative), source, fold_seed0..2 (stratified 5-fold assignment used in the paper), suspected_noisy (model-flagged, not verified), llm_audit_label (P/N/U from a single LLM annotator for 200 reviews; not ground truth).
