#!/bin/bash
cd .
while [ ! -f results/banglabook_result.json ]; do sleep 30; done
BB_MODEL=banglabert_large BB_TAG=_large BB_LR=2e-5 BB_BS=8 python3 -I src/cv_bert.py primary 1,2 4 > results/bert_large_primary_s12.log 2>&1
python3 -I src/cv_bert.py primary 1,2 4 > results/bert_primary_s12.log 2>&1
echo DONE > results/queue4.done
