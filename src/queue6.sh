#!/bin/bash
cd ./src
BK_MODEL=banglabert_large BK_LR=1e-5 BK_EPOCHS=3 BK_BS=16 BK_CWPOW=0   BK_SEED=1 BK_TAG=v4a python3 bert_banglabook_v4.py > ../results/bb_v4a.log 2>&1
BK_MODEL=banglabert_large BK_LR=1.5e-5 BK_EPOCHS=3 BK_BS=16 BK_CWPOW=0.5 BK_SEED=2 BK_TAG=v4b python3 bert_banglabook_v4.py > ../results/bb_v4b.log 2>&1
BK_MODEL=banglabert BK_LR=3e-5 BK_EPOCHS=3 BK_BS=32 BK_CWPOW=0.5 BK_SEED=3 BK_TAG=v4c python3 bert_banglabook_v4.py > ../results/bb_v4c.log 2>&1
touch ../results/queue6.done
