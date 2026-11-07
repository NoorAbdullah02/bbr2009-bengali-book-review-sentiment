#!/bin/bash
cd .
while [ ! -f results/queue6.done ]; do sleep 30; done
cd src
for m in xlmr_base:xlmr mbert:mbert; do
  M=${m%%:*}; T=${m##*:}
  BB_MODEL=$M python3 tune_bert.py $T > ../results/tune_$T.log 2>&1
  LR=$(python3 -c "import json;print(json.load(open('../results/tune_$T.json'))['best']['lr'])")
  EP=$(python3 -c "import json;print(json.load(open('../results/tune_$T.json'))['best']['epochs'])")
  BB_MODEL=$M BB_TAG=_${T}_tuned BB_LR=$LR BB_BS=16 python3 cv_bert.py primary 0,1,2 $EP > ../results/cv_${T}_tuned.log 2>&1
done
# ablation, 3 seeds
BB_TAG=_abl3_default python3 cv_bert.py primary 0,1,2 4 > ../results/abl3_default.log 2>&1
BB_TAG=_abl3_lr5e5 BB_LR=5e-5 python3 cv_bert.py primary 0,1,2 4 > ../results/abl3_lr.log 2>&1
BB_TAG=_abl3_nocw BB_CW=0 python3 cv_bert.py primary 0,1,2 4 > ../results/abl3_nocw.log 2>&1
BB_TAG=_abl3_len64 BB_MAXLEN=64 python3 cv_bert.py primary 0,1,2 4 > ../results/abl3_len.log 2>&1
BB_TAG=_abl3_ep2 python3 cv_bert.py primary 0,1,2 2 > ../results/abl3_ep2.log 2>&1
touch ../results/queue7.done
