import unittest

import pandas as pd
import numpy as np

import os

def is_same_splits(task):
    PATH = f"../predictions/{task}/"
    for file in os.listdir(PATH):
        if file.endswith('crosstrained.tsv'):
            continue
        if "final" in file:
            continue

        df = pd.read_csv(PATH + file, sep="\t")
        df1 = pd.read_csv(PATH + file.replace(".tsv", "_crosstrained.tsv"), sep="\t")
        for split in ['train', 'dev']:
            if not np.all(df[df['train/dev'] == split]['tweet_id'] == df1[df1['train/dev'] == split]['tweet_id']):
                return False
    return True


class TestSameSplits(unittest.TestCase):

    def test_task5(self):
        self.assertTrue(is_same_splits('task5'))

    def test_task6(self):
        self.assertTrue(is_same_splits('task6'))


if __name__ == '__main__':
    unittest.main()
