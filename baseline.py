
# coding: utf-8

# In[ ]:


from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn
from training.training import Trainer
import os.path as osp
import cv2
import numpy as np


import torch.nn.functional as F


# In[ ]:


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


# In[ ]:


# from fastai.conv_learner import *
# from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt


# In[ ]:


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


# In[ ]:


PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'


# In[ ]:


train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)


# In[ ]:


def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)


# In[ ]:


TARGET_SIZE=512


# In[ ]:


aug = Compose([
    HorizontalFlip(p=0.7),
    RandomGamma(p=0.7),
    GridDistortion(p=0.6),
    OpticalDistortion(p=0.6),
    ElasticTransform(p=0.6),
    Resize(height=TARGET_SIZE, width=TARGET_SIZE)
])

val_aug=Resize(height=TARGET_SIZE, width=TARGET_SIZE)


# In[ ]:


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


# In[ ]:


train_names, val_names = train_test_split(train_names)


# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()


# In[ ]:


THRESHOLD=0.0


# In[ ]:


loss = FocalLoss()


# In[ ]:


def mymetric(pred, target):
    preds = (pred > THRESHOLD).int()
    targs = target.int()
    return (preds==targs).float().mean()

def myloss(pred, target):
    return loss(pred, target)


# In[ ]:


model = resnet50(pretrained=True)
w = model.conv1.weight
model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
model.conv1.weight = torch.nn.Parameter(torch.cat((w,torch.mean(w,dim=1).unsqueeze(1)),dim=1))

model.avgpool = nn.Sequential(
    nn.MaxPool2d(kernel_size=6, stride=2,padding=0),
    nn.AvgPool2d(kernel_size=5, stride=2,padding=0)
)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.Dropout(0.5),
    nn.Linear(256, 28)
)

MODEL_NAME='resnet50'
BATCH_SIZE=24
DEVICE=0
EPOCHS=200

train_ds = ProteinDataset(train_names, TRAIN)
val_ds = ProteinDataset(val_names, TRAIN, val_aug)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(myloss, mymetric, optimizer, MODEL_NAME, None, DEVICE)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=BATCH_SIZE)


# In[ ]:


trainer.output_watcher = None

ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
model.to(DEVICE)


for i in range(EPOCHS):
    trainer.train(train_loader, model, i)
    trainer.validate(val_loader, model)


import pickle
pickle.dump(trainer, open('trainer.pkl','wb'))
