#!/bin/bash

git show -s --format=%H > ./logs/train_task6.log

time python train_models.py \
	--train-file ../data/2021/Subtask\ 6/train.tsv \
	--dev-file ../data/2021/Subtask\ 6/valid.tsv \
	--num-labels 3 \
	--learning-curve-points 10 50 100 175 250 500 750 1000 1500 2000 3000 4000 5000 6000 7000 8000  \
	--model-save-path ./models/task6 \
	--prediction-save-path ./predictions/task6  \
	--task 6 \
	>> ./logs/train_task6.log 2> ./logs/train_task6.err
