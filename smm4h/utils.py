import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score

import functools
print = functools.partial(print, flush=True)

class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

        def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        def __len__(self):
                return len(self.labels)


class Converter():
        def __init__(self, task):
                self.task = task

        def lbl2idx(self, x):
                if self.task == 6:
                        task6_lbl2idx = {'Lit-News_mentions': 0, 'Nonpersonal_reports': 1, 'Self_reports': 2}
                        return task6_lbl2idx[x]
                return int(x)

        def idx2lbl(self, x):
                if self.task == 6:
                        task6_idx2lbl = {0: 'Lit-News_mentions', 1: 'Nonpersonal_reports', 2: 'Self_reports'}
                        if type(x) == torch.Tensor:
                            x = x.item()
                        return task6_idx2lbl[x]
                return int(x)


def train_model(model, train_dataset):
        training_args = TrainingArguments(
                output_dir='./results',  # output directory
                num_train_epochs=3,  # total number of training epochs
                per_device_train_batch_size=64,  # batch size per device during training
                per_device_eval_batch_size=64,  # batch size for evaluation
                warmup_steps=500,  # number of warmup steps for learning rate scheduler
                weight_decay=0.01,  # strength of weight decay
                logging_dir='./logs',  # directory for storing logs
                logging_steps=10,
        )

        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=train_dataset,
        )

        trainer.train()
        return trainer

def display_f1(name, df):
        train_df = df[df['train/dev'] == 'train']
        dev_df = df[df['train/dev'] == 'dev']
        print(f'Model: {name}')
        print(f'\ttrain f1: {f1_score(train_df["label"], train_df["predicted-label"], average="micro")}')
        print(f'\tdev   f1: {f1_score(dev_df["label"], dev_df["predicted-label"], average="micro")}')
