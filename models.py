from se_resnet import se_resnet152
from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn
from training.training import Trainer
import os.path as osp
import cv2
import numpy as np
from tqdm import *

def get_resnet152():    
    model = resnet152(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w,torch.mean(w,dim=1).unsqueeze(1)),dim=1))

    model.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=6, stride=2,padding=0),
        nn.AvgPool2d(kernel_size=5, stride=2,padding=0)
    )
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 28))

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('resnet152_best.pth.tar'))
    return model

def get_se_resnet152():
    model = se_resnet152(num_classes=1000)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w,torch.mean(w,dim=1).unsqueeze(1)),dim=1))

    model.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=6, stride=2,padding=0),
        nn.AvgPool2d(kernel_size=5, stride=2,padding=0)
    )
    model.fc = nn.Linear(model.fc.in_features, 28)
    
    model = nn.DataParallel(model)
    #model.load_state_dict(torch.load('se_resnet152_best.pth.tar'))

    return model

def get_model(name):
    if name == 'resnet152':
        return get_resnet152()
    elif name == 'se_resnet152.yaml':
        return get_se_resnet152()
    else: raise Exception('not supported model')

