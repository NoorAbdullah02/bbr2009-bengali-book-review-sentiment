# BanglaBook official split (Kabir et al., ACL Findings 2023, CC BY-NC-SA 4.0)
The raw CSV files are stored as gzip parts to keep each upload small. Rebuild them from the repository root with:

    cat data/banglabook_parts/train.csv.gz.part* | gunzip > data/banglabook/train.csv
    cat data/banglabook_parts/validation.csv.gz.part* | gunzip > data/banglabook/validation.csv
    cat data/banglabook_parts/test.csv.gz.part* | gunzip > data/banglabook/test.csv

(create `data/banglabook/` first). The same data is public at https://github.com/Starscream-11813/BanglaBook .
