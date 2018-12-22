from torchvision import models
from pretrainedmodels.models import bninception, se_resnext101_32x4d
from torch import nn
from kaggle_human_protein_baseline.config import config
from collections import OrderedDict
import torch.nn.functional as F


def get_net():
    model = bninception(pretrained="imagenet")
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, config.num_classes),
    )
    return model


def get_resnet152():
    model = se_resnext101_32x4d()
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

    model.avg_pool = nn.AdaptiveAvgPool2d(1)

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, config.num_classes),
    )
    return model

