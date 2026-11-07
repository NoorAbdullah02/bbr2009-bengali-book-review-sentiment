#!/bin/bash
cd .
python3 -I src/cv_bert.py mendeley 0,1,2 4 > results/bert_mendeley.log 2>&1
python3 -I src/cv_bert.py rokomari 0 4 > results/bert_rokomari.log 2>&1
echo QUEUE_DONE > results/queue.done
