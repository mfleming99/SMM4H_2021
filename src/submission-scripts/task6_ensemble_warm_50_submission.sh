python3 make_ensamble_submission.py \
	--model-paths \
	./models/task6/50_kfold-split=0_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task6/50_kfold-split=1_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task6/50_kfold-split=2_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task6/50_kfold-split=3_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task6/50_kfold-split=4_model-name=DistilBertForSequenceClassification_crosstrained \
	--test-file ../data/2021/Subtask\ 6/test_tweets.tsv \
	--task 6 \
	--output-file submissions/task6/ensemble_warm_50
