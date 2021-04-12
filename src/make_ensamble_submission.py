import torch
import argparse
import pandas as pd
import numpy as np
import os
import zipfile
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from scipy import stats

from smm4h import TaskDataset, Converter

parser = argparse.ArgumentParser(description='Make a valid submission file of ensembled predictions to submit to CodaLabs')
parser.add_argument('--model-paths', nargs='*', help='list of paths to models to make predictions with')
parser.add_argument('--test-file', help='file to make predictions on')
parser.add_argument('--output-file', help='file to write predictions to')
parser.add_argument('--task', type=int, help='which task is this running')
args = parser.parse_args()

converter = Converter(args.task)

def get_preds(model, test_dataset):
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    outputs = []

    for batch in loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        with torch.no_grad():
            predictions = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)
            outputs += list(map(converter.idx2lbl, predictions))
    return outputs


if __name__ == '__main__':
    predictions = []
    test_df = pd.read_csv(args.test_file, sep='\t')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_encodings = tokenizer(list(test_df['tweet']), truncation=True, padding=True)
    test_dataset = TaskDataset(test_encodings, test_df['tweet'].map(lambda x: -1))
    for model_file in args.model_paths:
        model = DistilBertForSequenceClassification.from_pretrained(model_file, num_labels=2 if args.task == 5 else 3).cuda()
        model.eval()
        predictions.append(get_preds(model, test_dataset))
    majority_vote = stats.mode(predictions).mode[0]
    test_df['label'] = majority_vote
    with open(f'{args.output_file}.tsv', 'w+') as f:
        test_df.to_csv(f, sep='\t', index=False)
    zipfile.ZipFile(f'{args.output_file}.zip', mode='w').write(f'{args.output_file}.tsv', f'{args.output_file}.tsv'.split('/')[-1])

    
