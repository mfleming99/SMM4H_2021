import torch
import argparse
import pandas as pd
import numpy as np
import os
import zipfile
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from smm4h import TaskDataset, Converter

parser = argparse.ArgumentParser(description='Make a valid zip file to submit to CodaLabs')
parser.add_argument('--model-path', help='path to model to make predictions with')
parser.add_argument('--test-file', help='file to make predictions on')
parser.add_argument('--output-file', help='file to write predictions to')
parser.add_argument('--task', type=int, help='which task is this running')
args = parser.parse_args()

converter = Converter(args.task)

def create_submission():
    test_df = pd.read_csv(args.test_file, sep='\t')

    model = DistilBertForSequenceClassification.from_pretrained(args.model_path).cuda()
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    test_encodings = tokenizer(list(test_df['tweet']), truncation=True, padding=True)
    test_dataset = TaskDataset(test_encodings, test_df['tweet'].map(lambda x: -1))

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    outputs = []

    for batch in loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        with torch.no_grad():
            predictions = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)
            outputs += list(map(converter.idx2lbl, predictions))

    test_df['label'] = outputs
    with open(f'{args.output_file}.tsv', 'w+') as f:
        test_df.to_csv(f, sep='\t', index=False)
    zipfile.ZipFile(f'{args.output_file}.zip', mode='w').write(f'{args.output_file}.tsv', f'{args.output_file}.tsv'.split('/')[-1])


if __name__ == '__main__':
    create_submission()
