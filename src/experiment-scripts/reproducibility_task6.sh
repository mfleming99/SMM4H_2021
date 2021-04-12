#!/bin/bash

git show -s --format=%H > ./logs/reproducibility_task6.log

python3 model_reproducibility.py \
	--models-dir ./adam-final-models/models/task6 \
	--predictions-dir predictions/task6 \
	--task 6 \
	>> ./logs/reproducibility_task6.log
