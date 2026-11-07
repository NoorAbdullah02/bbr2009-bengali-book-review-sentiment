#!/bin/bash
cd .
BB_MODEL=xlmr_base BB_TAG=_xlmr BB_LR=2e-5 BB_BS=16 python3 -I src/cv_bert.py primary 0,1,2 4 > results/cv_xlmr.log 2>&1
for i in $(seq 1 60); do [ -f models/mbert/model.safetensors ] && break; sleep 30; done
BB_MODEL=mbert BB_TAG=_mbert BB_LR=3e-5 BB_BS=16 python3 -I src/cv_bert.py primary 0,1,2 4 > results/cv_mbert.log 2>&1
touch results/queue5.done
