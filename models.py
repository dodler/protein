from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet152, resnet34
import os.path as osp

from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154
from se_resnet import se_resnet152


def load_if_exists(model, path):
    '''
    loads state for model is state exists on the disk
    :param model:
    :param path: path to the saved state
    :return:
    '''
    print('loading path:' + path)
    if osp.exists(path):
        print('path ok')
        model.load_state_dict(torch.load(path))
    else:
        print('path not found')


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
    load_if_exists(model, 'resnet152_best.pth.tar')
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
    load_if_exists(model, 'se_resnet152_best.pth.tar')

    return model


def get_se_resnext50():
    model = se_resnext50_32x4d()
    inplanes = 64
    conv0 = nn.Conv2d(4, inplanes, kernel_size=7, stride=2,
                      padding=3, bias=False)
    w = model.layer0[0].weight
    conv0.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    layer0_modules = [('conv1', conv0), ('bn1', nn.BatchNorm2d(inplanes)),
                      ('relu1', nn.ReLU(inplace=True)), ('pool', nn.MaxPool2d(3, stride=2,
                                                                              ceil_mode=True))]
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.avg_pool = nn.Sequential(
        nn.AvgPool2d(11, stride=1),
        nn.Dropout2d(0.3),
        nn.AvgPool2d(4, stride=2),
    )

    model.last_linear = nn.Sequential(
        nn.Linear(8192, 1024),
        nn.Dropout(),
        nn.Linear(1024, 28)
    )

    model = nn.DataParallel(model)
    load_if_exists(model, 'se_resnext50_best.pth.tar')
    return model


def get_senet154():
    model = senet154(pretrained=None)

    inplanes = 64
    layer0_modules = [
        ('conv1', nn.Conv2d(4, 64, 3, stride=2, padding=1,
                            bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                            bias=False)),
        ('bn2', nn.BatchNorm2d(64)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                            bias=False)),
        ('bn3', nn.BatchNorm2d(inplanes)),
        ('relu3', nn.ReLU(inplace=True)),
    ]
    layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                ceil_mode=True)))

    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.avg_pool = nn.Sequential(
        nn.AvgPool2d(7, stride=1),
        nn.Dropout2d(0.3),
        nn.AvgPool2d(7, stride=1)
    )

    model.last_linear = nn.Sequential(
        nn.Linear(model.last_linear.in_features, 768),
        nn.Dropout(),
        nn.Linear(768, 28)
    )
    model = nn.DataParallel(model)
    load_if_exists(model, 'senet154_best.pth.tar')
    return model


def get_resnet34():
    model = resnet34(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    w = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))
    model.conv1.weight = w

    model.avgpool = nn.Sequential(
        nn.AvgPool2d(7,stride=1),
        nn.Dropout2d(),
        nn.AvgPool2d(7,stride=1)
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 768),
        nn.Dropout(),
        nn.Linear(768, 28)
    )

    return model


mdl2name = {
    'resnet152': get_resnet152,
    'se_resnet152': get_se_resnet152,
    'se_resnext50': get_se_resnext50,
    'senet154': get_senet154,
    'resnet34': get_resnet34
}


def get_model(name):
    if name in mdl2name.keys():
        return mdl2name[name]()
    else:
        raise Exception('not supported model')
