#!/bin/bash
cd ./src
BK_MODEL=banglabert_large BK_LR=1e-5   BK_EPOCHS=3 BK_BS=16 BK_CWPOW=0    BK_SEED=4 BK_TAG=v4d python3 bert_banglabook_v4.py > ../results/bb_v4d.log 2>&1
BK_MODEL=banglabert_large BK_LR=1.2e-5 BK_EPOCHS=3 BK_BS=16 BK_CWPOW=0.25 BK_SEED=5 BK_TAG=v4e python3 bert_banglabook_v4.py > ../results/bb_v4e.log 2>&1
BK_MODEL=banglabert_large BK_LR=1e-5   BK_EPOCHS=3 BK_BS=16 BK_CWPOW=0.5  BK_SEED=6 BK_TAG=v4f python3 bert_banglabook_v4.py > ../results/bb_v4f.log 2>&1
touch ../results/queue8.done
