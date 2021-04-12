#!/bin/bash

git show -s --format=%H > ./logs/train_task5_final.log

time python train_models.py \
	--train-file ../data/2021/Subtask\ 5/train.tsv \
	--dev-file ../data/2021/Subtask\ 5/valid.tsv \
	--num-labels 3 \
	--model-save-path ./models/task5 \
	--prediction-save-path ./predictions/task5  \
	--task 5 \
	--final \
	>> ./logs/train_task5_final.log 2> ./logs/logs_train_task5_final.err
