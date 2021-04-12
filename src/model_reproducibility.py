import torch
import argparse
import pandas as pd
import numpy as np
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from smm4h import TaskDataset, Converter

parser = argparse.ArgumentParser(description='Assert a set of models can reproduce their predictions')
parser.add_argument('--models-dir', help='directory of models used to generate predictions specified in --predictions-dir')
parser.add_argument('--predictions-dir', help='directory of predictions to check')
parser.add_argument('--task', type=int, help='which task this is running')
args = parser.parse_args()

converter = Converter(args.task)

import functools
print = functools.partial(print, flush=True)

def assert_reproducibility():
    prediction_files = os.listdir(args.predictions_dir)
    prediction_files.sort()

    model_files = os.listdir(args.models_dir)
    model_files.sort()

    for pred_file, model_file in zip(prediction_files, model_files):
        preds = pd.read_csv(f'{args.predictions_dir}/{pred_file}', sep='\t')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(f'{args.models_dir}/{model_file}', num_labels= 2 if args.task == 5 else 3).cuda()

        encodings = tokenizer(list(preds['tweet']), truncation=True, padding=True)
        dataset = TaskDataset(encodings, preds['tweet'].map(lambda x: -1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        outputs = []

        for batch in loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            with torch.no_grad():
                predictions = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)


            outputs += list(map(converter.idx2lbl, predictions))

        print(f'asserting match between \n\t model: {model_file} \n\t predictions: {pred_file}')
        if np.array_equal(list(preds['predicted-label']), outputs):
            print('perfect match')
        else:
            print(f'warning there are {len(outputs) - sum(pd.Series(list(preds["predicted-label"])) == pd.Series(outputs))} non-matching predictions')


if __name__ == '__main__':
    assert_reproducibility()
