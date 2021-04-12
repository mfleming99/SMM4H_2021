#!/bin/bash

git show -s --format=%H > ./logs/cross_train_task5.log

time python cross_train_models.py \
	--train-file ../data/2021/Subtask\ 5/train.tsv \
	--dev-file ../data/2021/Subtask\ 5/valid.tsv \
	--num-labels 2 \
	--learning-curve-points 10 50 100 175 250 500 750 1000 1500 2000 3000 4000 5000 6000 \
	--model-save-path  ./models/task5 \
	--prediction-save-path ./predictions/task5  \
	--task 5 \
	--model-load-path ./models/task6/9452_final_model-name=DistilBertForSequenceClassification \
	>> ./logs/cross_train_task5.log 2> ./logs/cross_train_task5.err
