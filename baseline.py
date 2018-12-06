# coding: utf-8
import gc
import pickle

import cv2
import numpy as np
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
    Resize)
from torchvision.models.resnet import resnet152
from training.training import Trainer

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from se_resnet import se_resnet152
from utils import name_label_dict

torch.manual_seed(42)
np.random.seed(42)

PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'

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

aug = Compose([
    HorizontalFlip(p=0.7),
    RandomGamma(p=0.7),
    GridDistortion(p=0.6),
    OpticalDistortion(p=0.6),
    ElasticTransform(p=0.6),
    Resize(height=TARGET_SIZE, width=TARGET_SIZE)
])

val_aug = Resize(height=TARGET_SIZE, width=TARGET_SIZE)


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


train_names, val_names = train_test_split(train_names)


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


THRESHOLD = 0.0
loss = FocalLoss()


def mymetric(pred, target):
    preds = (pred > THRESHOLD).int()
    targs = target.int()
    return (preds == targs).float().mean()


def myloss(pred, target):
    return loss(pred, target)


def get_resnet152():
    model = resnet152(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    model.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=6, stride=2, padding=0),
        nn.AvgPool2d(kernel_size=5, stride=2, padding=0)
    )
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 28))

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('resnet152_best.pth.tar'))
    return model


def get_se_resnet152():
    model = se_resnet152(num_classes=1000)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    model.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=6, stride=2, padding=0),
        nn.AvgPool2d(kernel_size=5, stride=2, padding=0)
    )
    model.fc = nn.Linear(model.fc.in_features, 28)

    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('se_resnet152_best.pth.tar'))

    return model


def get_model(name):
    if name == 'resnet152':
        return get_resnet152()
    elif name == 'se_resnet152':
        return get_se_resnet152()
    else:
        raise Exception('not supported model')


MODEL_NAME = 'se_resnet152'
BATCH_SIZE = 10
DEVICE = 0
EPOCHS = 100

model = get_model(MODEL_NAME)

train_ds = ProteinDataset(train_names, TRAIN)
val_ds = ProteinDataset(val_names, TRAIN, val_aug)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(myloss, mymetric, optimizer, MODEL_NAME, None, DEVICE)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

trainer.output_watcher = None
model.to(DEVICE)

for i in range(EPOCHS):
    trainer.train(train_loader, model, i)
    trainer.validate(val_loader, model)
    trainer.full_history = None
    gc.collect()
    trainer.full_history = {}

pickle.dump(trainer, open('trainer.pkl', 'wb'))
