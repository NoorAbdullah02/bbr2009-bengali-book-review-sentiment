# BBR-2009 datasheet
Binary (0 negative / 1 positive) Bengali book-review sentiment, 2009 unique reviews: 1029 negative, 980 positive.
Sources: Mendeley Bengali Book Reviews (doi 10.17632/2wcw3sxxr3.1, CC BY 4.0; manual labels) and BanglaBook (Kabir et al., ACL Findings 2023, CC BY-NC-SA 4.0; labels derived from star ratings).
Per source: {'mendeley': 1414, 'banglabook': 595}.
Columns: id, text, label, source, fold_seed0..2 (stratified 5-fold assignment used in the paper), suspected_noisy (model-flagged, NOT manually verified).
Known issues: BanglaBook rows are ~95% negative, so source and label are confounded; rating-derived labels are noisy.
Licence of this release: CC BY-NC-SA 4.0 (inherits the most restrictive source licence). Non-commercial use only.

llm_audit_label: blind label (P/N/U) from a single LLM annotator for 200 reviews (69 suspected + 131 random); not a human annotation and not ground truth.
