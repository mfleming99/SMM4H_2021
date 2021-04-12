#!/bin/bash

git show -s --format=%H > ./logs/reproducibility_task5.log

python3 model_reproducibility.py \
	--models-dir ./models/task5 \
	--predictions-dir predictions/task5 \
	--task 5 \
	>> ./logs/reproducibility_task5.log
