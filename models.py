from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet152
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
    w = conv0.weight
    conv0.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    layer0_modules = [('conv1', conv0), ('bn1', nn.BatchNorm2d(inplanes)),
                      ('relu1', nn.ReLU(inplace=True)), ('pool', nn.MaxPool2d(3, stride=2,
                                                                              ceil_mode=True))]
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.last_linear = nn.Sequential(
        nn.Linear(model.last_linear.in_features, 768),
        nn.Dropout(),
        nn.Linear(768, 28)
    )

    model = nn.DataParallel(model)
    load_if_exists(model, 'se_resnext50_best.pth.tar')
    return model


def get_senet154():
    model = senet154(pretrained=None)

    model.last_linear = nn.Sequential(
        nn.Linear(model.last_linear.in_features, 768),
        nn.Dropout(),
        nn.Linear(768, 28)
    )
    model = nn.DataParallel(model)
    load_if_exists(model, 'senet154_best.pth.tar')
    return model


mdl2name = {
    'resnet152': get_resnet152,
    'se_resnet152': get_se_resnet152,
    'se_resnext50': get_se_resnext50,
    'senet154': get_senet154,
}


def get_model(name):
    if name in mdl2name.keys():
        return mdl2name[name]()
    else:
        raise Exception('not supported model')
