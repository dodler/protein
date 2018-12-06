import cv2
import numpy as np
import torch
from albumentations import (
    Resize)
from models import get_model
from prot_dataset import ProteinDataset

torch.manual_seed(42)
np.random.seed(42)
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import *

TARGET_SIZE=512

PATH = './'
TRAIN = '/root/data/protein/train/'
TEST = '/root/data/protein/test/'
LABELS = '/root/data/protein/train.csv'
SAMPLE = '/root/data/protein/sample_submission.csv'


train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)

def open_rgby(path,id): #a function that reads RGBY image
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id+'_'+color+'.png'), flags).astype(np.float32)/255
           for color in colors]
    return np.stack(img, axis=-1)

val_aug=Resize(height=TARGET_SIZE, width=TARGET_SIZE)

MODEL_NAME='se_resnet152'
BATCH_SIZE=10
DEVICE=0
EPOCHS=100
WORKERS = 10

model = get_model(MODEL_NAME).to(DEVICE)
subm = pd.read_csv(SAMPLE)
test_ds = ProteinDataset(subm.Id.values, TEST, val_aug)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

result = []
for batch_idx, (input, target) in tqdm(enumerate(test_loader)):
    input = input.to(DEVICE)
    result.append(model(input).detach().cpu())
    
preds = []
for r in result:
    for t in r:
        preds.append(torch.sigmoid(t).numpy())
        
THRESHOLD = 0.3

for i in tqdm(range(subm.Id.size)):
    subm.iloc[i,1] = ' '.join(np.where(preds[i] > THRESHOLD)[0].astype(str))
    
print(np.array([len(k) for k in subm.Predicted.str.split()]).mean())
subm.to_csv('my_subm.csv',index=False)
print('done')
