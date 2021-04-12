#!/bin/bash

git show -s --format=%H > ./logs/train_task6_final.log

time python train_models.py \
	--train-file ../data/2021/Subtask\ 6/train.tsv \
	--dev-file ../data/2021/Subtask\ 6/valid.tsv \
	--num-labels 3 \
	--model-save-path ./models/task6 \
	--prediction-save-path ./predictions/task6  \
	--task 6 \
	--final \
	>> ./logs/train_task6_final.log 2> ./logs/logs_train_task6_final.err
