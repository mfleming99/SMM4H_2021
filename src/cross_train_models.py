import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.model_selection import KFold
from smm4h import TaskDataset, Converter, train_model, display_f1

parser = argparse.ArgumentParser(description='Train models (either kfold or final) for a specified task given a model to warmstart weights with')
parser.add_argument('--train-file', help="path to train file")
parser.add_argument('--dev-file', help='path to dev/val file')
parser.add_argument('--num-labels', type=int, help='number of labels for the task')
parser.add_argument('--task', type=int, help='which task this is running')

parser.add_argument('--learning-curve-points', nargs='*', type=int, help='list of number of examples to train and predict on')
parser.add_argument('--model-save-path', help='directory to save model')
parser.add_argument('--prediction-save-path', help='directory to save predictions')

parser.add_argument('--model-load-path', help='path to model to warmstart the model weights with')

parser.add_argument('--final', default=False, action='store_true', help='set to train a model on both train/dev')
args = parser.parse_args()

converter = Converter(args.task)


def k_fold_train_and_write(data):
    kf = KFold(n_splits=5, shuffle=True)
    split = 0
    for train_idx, test_idx in kf.split(data):

        train = data.loc[train_idx, :].reset_index(drop=True)
        test = data.loc[test_idx, :].reset_index(drop=True)

        for n in args.learning_curve_points:

            model = DistilBertForSequenceClassification.from_pretrained(args.model_load_path)
            model.num_labels = args.num_labels
            model.classifier = torch.nn.Linear(768, args.num_labels)
            
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            train_encodings = tokenizer(list(train.head(n)['tweet']), truncation=True, padding=True)
            test_encodings = tokenizer(list(test['tweet']), truncation=True, padding=True)

            train_dataset = TaskDataset(train_encodings, train.head(n)['label'].map(converter.lbl2idx))
            test_dataset = TaskDataset(test_encodings, test['label'].map(lambda x: 0))
            trainer = train_model(model, train_dataset)
            trainer.model.save_pretrained(f'{args.model_save_path}/{len(train.head(n))}_kfold-split={split}_model-name={type(model).__name__}_crosstrained')

            train_predictions = trainer.predict(train_dataset).predictions.argmax(-1)
            test_predictions = trainer.predict(test_dataset).predictions.argmax(-1)

            train_head = train.head(n)
            train_head['train/dev'] = 'train'
            train_head['predicted-label'] = train_predictions
            test['train/dev'] = 'dev'
            test['predicted-label'] = test_predictions
            df = train_head.append(test)


            df['predicted-label'] = df['predicted-label'].map(converter.idx2lbl)
            name = '\t'.join([type(model).__name__, str(split), str(n), 'crosstrained'])
            display_f1(name, df)

            with open(f'{args.prediction_save_path}/{len(train.head(n))}_kfold-split={split}_model-name={type(model).__name__}_crosstrained.tsv','w+') as f:
                df.drop(['index'], axis=1).to_csv(f, sep='\t', index=False)

        split += 1
        
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_df = pd.read_csv(args.train_file, sep='\t')
    dev_df = pd.read_csv(args.dev_file, sep='\t')
    df = pd.concat([train_df, dev_df]).drop_duplicates(subset=['tweet']).reset_index()

    if not args.final:
        k_fold_train_and_write(df)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        train_encodings = tokenizer(list(df['tweet']), truncation=True, padding=True)
        train_dataset = TaskDataset(train_encodings, df['label'].map(converter.lbl2idx))

        model = DistilBertForSequenceClassification.from_pretrained(args.model_load_path)
        model.num_labels = args.num_labels
        model.classifier = torch.nn.Linear(768, args.num_labels)
        
        trainer = train_model(
            model,
            train_dataset
        )

        trainer.model.save_pretrained(f'{args.model_save_path}/{len(df)}_final_model-name={type(trainer.model).__name__}_crosstrained')
        train_predictions = trainer.predict(train_dataset).predictions.argmax(-1)
        df['train/dev'] = 'train'
        df['predicted-label'] = train_predictions
        df['predicted-label'] = df['predicted-label'].map(converter.idx2lbl)
        with open(f'{args.prediction_save_path}/{len(df)}_final_model-name={type(trainer.model).__name__}_crosstrained.tsv','w+') as f:
            df.drop(['index'], axis=1).to_csv(f, sep='\t', index=False)
