#!/bin/bash
cd ./src
while [ ! -f ../results/queue2.done ]; do sleep 20; done
PYTHONPATH=./src python3 explain.py > ../results/explain.log 2>&1
echo DONE > ../results/queue3.done
