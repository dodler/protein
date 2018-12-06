# coding: utf-8
from se_resnet import se_resnet152
from torchvision.models.resnet import resnet50, resnet152
import torch
import torch.nn as nn
from training.training import Trainer
import os.path as osp
import cv2
import numpy as np
import gc
import pickle

import torch.nn.functional as F
from tqdm import *



name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }


from albumentations import (
    VerticalFlip,
    HorizontalFlip,
    Compose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    Resize)
torch.manual_seed(42)
np.random.seed(42)


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'





def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)





TARGET_SIZE=512

aug=Resize(height=TARGET_SIZE, width=TARGET_SIZE)





class ProteinDataset:
    def __init__(self, names, path,aug=aug):
        self.names=names
        self.aug=aug
        self.path=path
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        if(self.path == TEST): label= np.zeros(len(name_label_dict),dtype=np.int)
        else:
            labels = self.labels.loc[self.names[idx]]['Target']
            label=np.eye(len(name_label_dict),dtype=np.float)[labels].sum(axis=0)
        
        img = open_rgby(self.path, self.names[idx])
        img = aug(image=img)['image']
        
        return torch.from_numpy(
            img
        ).permute([2,0,1]), torch.from_numpy(label).float()
