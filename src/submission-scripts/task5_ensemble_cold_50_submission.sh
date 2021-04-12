python3 make_ensamble_submission.py \
	--model-paths \
	./models/task5/50_kfold-split=0_model-name=DistilBertForSequenceClassification \
	./models/task5/50_kfold-split=1_model-name=DistilBertForSequenceClassification \
	./models/task5/50_kfold-split=2_model-name=DistilBertForSequenceClassification \
	./models/task5/50_kfold-split=3_model-name=DistilBertForSequenceClassification \
	./models/task5/50_kfold-split=4_model-name=DistilBertForSequenceClassification \
	--test-file ../data/2021/Subtask\ 5/test.tsv \
	--task 5 \
	--output-file submissions/task5/ensemble_cold_50
