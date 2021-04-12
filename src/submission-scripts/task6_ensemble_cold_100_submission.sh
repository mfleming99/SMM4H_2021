python3 make_ensamble_submission.py \
	--model-paths \
	./adam-final-models/models/task6/100_kfold-split=0_model-name=DistilBertForSequenceClassification \
	./adam-final-models/models/task6/100_kfold-split=1_model-name=DistilBertForSequenceClassification \
	./adam-final-models/models/task6/100_kfold-split=2_model-name=DistilBertForSequenceClassification \
	./adam-final-models/models/task6/100_kfold-split=3_model-name=DistilBertForSequenceClassification \
	./adam-final-models/models/task6/100_kfold-split=4_model-name=DistilBertForSequenceClassification \
	--test-file ../data/2021/Subtask\ 6/test_tweets.tsv \
	--task 6 \
	--output-file submissions/task6/ensemble_cold_100
