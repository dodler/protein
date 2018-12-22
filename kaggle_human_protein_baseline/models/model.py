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
    model = se_resnext101_32x4d(pretrained='imagenet')
    inplanes = 64
    layer0_modules = [
        ('conv1', nn.Conv2d(4, inplanes, kernel_size=7, stride=2,
                            padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(inplanes)),
        ('relu1', nn.ReLU(inplace=True)),
    ]
    # To preserve compatibility with Caffe weights `ceil_mode=True`
    # is used instead of `padding=1`.
    layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                ceil_mode=True)))
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.avg_pool = nn.AdaptiveAvgPool2d(1)

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, config.num_classes),
    )
    return model

