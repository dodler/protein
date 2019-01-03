from collections import OrderedDict

import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
from pretrainedmodels import resnet18, resnet152, resnet34

from pretrainedmodels.models.senet import se_resnext50_32x4d, senet154

from models.densenet import DenseNet3
from models.wide_resnet import Wide_ResNet
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


def get_resnext50():
    model = resnext50(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))


def get_resnet152():
    model = resnet152(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    model.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=16, stride=2, padding=0),
    )
    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(model.fc.in_features, 28))

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

    return model


def get_se_resnext50():
    model = se_resnext50_32x4d()
    inplanes = 64
    w = model.layer0[0].weight
    conv0 = nn.Conv2d(4, inplanes, kernel_size=15, stride=2,
                      padding=3, bias=False)
    new_w = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))
    print(new_w.shape)
    conv0.weight = new_w

    layer0_modules = [('conv1', conv0), ('bn1', nn.BatchNorm2d(inplanes)),
                      ('relu1', nn.ReLU(inplace=True)), ('pool', nn.MaxPool2d(3, stride=2,
                                                                              ceil_mode=True))]
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.avg_pool = nn.AvgPool2d(16, 2)

    model.last_linear = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 28),
    )

    # model = nn.DataParallel(model)
    # load_if_exists(model, 'se_resnext50_best.pth.tar')
    return model


def get_se_resnet50():
    model = se_resnet50()
    inplanes = 64
    layer0_modules = [
        ('conv1', nn.Conv2d(4, inplanes, kernel_size=7, stride=2,
                            padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(inplanes)),
        ('relu1', nn.ReLU(inplace=True)),
    ]
    # To preserve compatibility with Caffe weights `ceil_mode=True`
    # is used instead of `padding=1`.
    layer0_modules.append(('pool', nn.MaxPool2d(4, stride=2,
                                                ceil_mode=True)))
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    model.avg_pool = nn.AvgPool2d(16, stride=1)
    model.last_linear = nn.Sequential(
        nn.Dropout(),
        nn.Linear(2048, 28)
    )

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
        nn.AvgPool2d(7, stride=1),
        nn.Dropout2d(),
        nn.AvgPool2d(7, stride=1)
    )

    model.fc = nn.Sequential(
        nn.Linear(8192, 2048),
        nn.Dropout(),
        nn.Linear(2048, 28)
    )

    return model


def get_densenet121():
    model = DenseNet3(num_classes=28, depth=10)
    model.conv1 = nn.Conv2d(4, 24, kernel_size=3, stride=1,
                            padding=1, bias=False)

    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(1728, 28)
    )

    return model


from pretrainedmodels.models import bninception


def get_bn_inception():
    model = bninception()
    conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

    w = model.conv1_7x7_s2.weight
    conv1_7x7_s2.weight = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    model.conv1_7x7_s2 = conv1_7x7_s2
    model.global_pool = nn.AvgPool2d(16, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
    model.last_linear = nn.Sequential(
        nn.Dropout(),
        nn.Linear(1024, 28)
    )

    return model


def get_resnet18():
    model = resnet18()
    w = model.conv1.weight
    w = torch.nn.Parameter(torch.cat((w, torch.mean(w, dim=1).unsqueeze(1)), dim=1))

    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.conv1.weight = w

    model.avgpool = nn.Sequential(
        nn.AvgPool2d(9, stride=1),
        nn.AvgPool2d(7, stride=1)
    )

    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(2048, 28)
    )

    return model


def get_wide_resnet28_10():
    model = Wide_ResNet()
    return model


mdl2name = {
    'resnet152': get_resnet152,
    'resnet18': get_resnet18,
    'se_resnet152': get_se_resnet152,
    'se_resnext50': get_se_resnext50,
    'senet154': get_senet154,
    'resnet34': get_resnet34,
    'densenet121': get_densenet121,
    'se_resnet50': get_se_resnet50,
    'bn_inception': get_bn_inception,
    'wide_resnet28_10': get_wide_resnet28_10,
}


def get_model(name):
    if name in mdl2name.keys():
        return mdl2name[name]()
    else:
        raise Exception('not supported model')
