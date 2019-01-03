# coding: utf-8
import collections
import os
import sys

from sklearn.metrics import f1_score
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomGamma,
    Resize, Rotate, Transpose, VerticalFlip)
from sklearn.model_selection import train_test_split
from catalyst.dl.runner import ClassificationRunner

import collections
from catalyst.dl.callbacks import (
    ClassificationLossCallback,
    Logger, TensorboardLogger,
    OptimizerCallback, SchedulerCallback, CheckpointCallback,
    PrecisionCallback, OneCycleLR)
from models.model_factory import get_model

from utils import name_label_dict, parse_config, F1ScoreCallback

torch.manual_seed(42)
np.random.seed(42)

PATH = '/home/kaggleprotein/lyan/data/'
TRAIN = PATH + 'protein/train/'
TEST = PATH + 'protein/test/'
LABELS = PATH + 'protein/train.csv'
SAMPLE = PATH + 'protein/sample_submission.csv'

PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'

THRESHOLD = 0.0

train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)


def open_rgby(path, id):  # a function that reads RGBY image
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


TARGET_SIZE = 512

train_aug = Compose([
    HorizontalFlip(p=0.7),
    VerticalFlip(p=0.7),
    Transpose(p=0.7),
    Rotate(30, p=0.7),
    RandomGamma(p=0.7),
    # Resize(height=TARGET_SIZE, width=TARGET_SIZE)
])

valid_aug = None  # Resize(height=TARGET_SIZE, width=TARGET_SIZE)


class ProteinDataset:
    def __init__(self, names, path, aug=train_aug):
        self.names = names
        self.aug = aug
        self.path = path
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        if self.path == TEST:
            label = np.zeros(len(name_label_dict), dtype=np.int)
        else:
            labels = self.labels.loc[self.names[idx]]['Target']
            label = np.eye(len(name_label_dict), dtype=np.float)[labels].sum(axis=0)

        img = open_rgby(self.path, self.names[idx])
        img = train_aug(image=img)['image']

        return torch.from_numpy(
            img
        ).permute([2, 0, 1]), torch.from_numpy(label).float()


train_names, valid_names = train_test_split(train_names)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


def mymetric(pred, target):
    preds = (pred > THRESHOLD).int()
    targs = target.int()
    return f1_score(targs.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')


if __name__ == '__main__':
    BATCH_SIZE = 16
    WORKERS = 6

    train_ds = ProteinDataset(train_names, TRAIN, train_aug)
    valid_ds = ProteinDataset(valid_names, TRAIN, valid_aug)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=WORKERS,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE,
                                               num_workers=WORKERS, pin_memory=True)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    model = get_model('se_resnext50')

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10, verbose=True)

    # the only tricky part
    n_epochs = 120
    # logdir = "/tmp/runs/"
    logdir = "/tmp/runs_se_resnext50/"

    callbacks = collections.OrderedDict()

    callbacks['f1_score'] = F1ScoreCallback()
    callbacks["loss"] = ClassificationLossCallback()
    callbacks["optimizer"] = OptimizerCallback()

    callbacks["scheduler"] = SchedulerCallback(reduce_metric="f1_score")

    callbacks["saver"] = CheckpointCallback()
    callbacks["logger"] = Logger()
    callbacks["tflogger"] = TensorboardLogger()

    runner = ClassificationRunner(
        model=model,
        criterion=FocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler)

    runner.train(
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        epochs=n_epochs, verbose=True)
