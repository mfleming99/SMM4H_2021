import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from smm4h import TaskDataset, Converter, train_model

import functools
print = functools.partial(print, flush=True)


parser = argparse.ArgumentParser(description='Train models (either kfold or final) for a specified task')

parser.add_argument('--train-file', help="path to train file")
parser.add_argument('--dev-file', help='path to dev/val file')
parser.add_argument('--num-labels', type=int, help='number of labels for the task')
parser.add_argument('--task', type=int, help='which task this is running')

args = parser.parse_args()

converter = Converter(args.task)


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_df = pd.read_csv(args.train_file, sep='\t')
    dev_df = pd.read_csv(args.dev_file, sep='\t')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(list(train_df['tweet']), truncation=True, padding=True)
    dev_encodings = tokenizer(list(dev_df['tweet']), truncation=True, padding=True)

    train_dataset = TaskDataset(train_encodings, train_df['label'].map(converter.lbl2idx))
    dev_dataset = TaskDataset(dev_encodings, dev_df['label'].map(lambda x: 0))
    trainer = train_model(
            DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=args.num_labels),
            train_dataset
    )
        
    dev_predictions = trainer.predict(dev_dataset).predictions.argmax(-1)
    print(f'dev f1: {f1_score(dev_df["label"].map(converter.lbl2idx), dev_predictions, average="micro" if args.task ==  6 else "binary")}')
