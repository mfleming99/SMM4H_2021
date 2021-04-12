python3 make_ensamble_submission.py \
	--model-paths \
	./adam-final-models/models/task6/250_kfold-split=0_model-name=DistilBertForSequenceClassification_crosstrained \
	./adam-final-models/models/task6/250_kfold-split=1_model-name=DistilBertForSequenceClassification_crosstrained \
	./adam-final-models/models/task6/250_kfold-split=2_model-name=DistilBertForSequenceClassification_crosstrained \
	./adam-final-models/models/task6/250_kfold-split=3_model-name=DistilBertForSequenceClassification_crosstrained \
	./adam-final-models/models/task6/250_kfold-split=4_model-name=DistilBertForSequenceClassification_crosstrained \
	--test-file ../data/2021/Subtask\ 6/test_tweets.tsv \
	--task 6 \
	--output-file submissions/task6/ensemble_warm_250
