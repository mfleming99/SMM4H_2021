python3 make_ensamble_submission.py \
	--model-paths \
	./models/task5/500_kfold-split=0_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task5/500_kfold-split=1_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task5/500_kfold-split=2_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task5/500_kfold-split=3_model-name=DistilBertForSequenceClassification_crosstrained \
	./models/task5/500_kfold-split=4_model-name=DistilBertForSequenceClassification_crosstrained \
	--test-file ../data/2021/Subtask\ 5/test.tsv \
	--task 5 \
	--output-file submissions/task5/ensemble_warm_500
