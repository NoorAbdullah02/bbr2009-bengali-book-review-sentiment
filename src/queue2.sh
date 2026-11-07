#!/bin/bash
cd .
while [ ! -f results/queue.done ]; do sleep 15; done
BB_MODEL=banglabert_large BB_TAG=_large BB_LR=2e-5 python3 -I src/cv_bert.py primary 0 4 > results/bert_large_primary.log 2>&1
BB_TAG=_abl_nocw BB_CW=0 python3 -I src/cv_bert.py primary 0 4 > results/abl_nocw.log 2>&1
BB_TAG=_abl_len64 BB_MAXLEN=64 python3 -I src/cv_bert.py primary 0 4 > results/abl_len64.log 2>&1
BB_TAG=_abl_lr5e5 BB_LR=5e-5 python3 -I src/cv_bert.py primary 0 4 > results/abl_lr5e5.log 2>&1
BB_TAG=_abl_ep2 python3 -I src/cv_bert.py primary 0 2 > results/abl_ep2.log 2>&1
echo DONE > results/queue2.done
