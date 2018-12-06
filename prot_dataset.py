# coding: utf-8
import cv2
import numpy as np
import torch
from albumentations import (
    Resize)

from utils import name_label_dict

torch.manual_seed(42)
np.random.seed(42)

import pandas as pd
import numpy as np
import os

PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'


def open_rgby(path, id):  # a function that reads RGBY image
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


TARGET_SIZE = 512

aug = Resize(height=TARGET_SIZE, width=TARGET_SIZE)


class ProteinDataset:
    def __init__(self, names, path, aug=aug):
        self.names = names
        self.aug = aug
        self.path = path
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        if (self.path == TEST):
            label = np.zeros(len(name_label_dict), dtype=np.int)
        else:
            labels = self.labels.loc[self.names[idx]]['Target']
            label = np.eye(len(name_label_dict), dtype=np.float)[labels].sum(axis=0)

        img = open_rgby(self.path, self.names[idx])
        img = aug(image=img)['image']

        return torch.from_numpy(
            img
        ).permute([2, 0, 1]), torch.from_numpy(label).float()
