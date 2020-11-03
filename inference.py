#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 07:46:22 2020

@author: Krishna
"""

import numpy as np
import glob
import torch
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
from utils import utils
#from models.uni_modal_ser_transformer_speech import UniModel
from cnn_spec_attention import AttnPooling
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
#### Dataset info


dataset_path_gujarathi = '/hdd2/Speech_data/MS_LID_challenge/TESTING/PARTA/Gujarati/Audio'
all_files = sorted(glob.glob(dataset_path_gujarathi+'/*.wav'))



def compute_features(audio_filepath):
    spec = utils.load_data(audio_filepath)
    return spec
    
    

def speech_collate(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        targets.append((sample['labels']))
    return specs, targets
 
### Data related
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda:0")
model = AttnPooling(num_classes=1).to(device)
model.load_state_dict(torch.load('model_checkpoint_gujarati/best_check_point_98_0.8220946915351507')['model'])
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss = nn.BCELoss()

csv_store='results/TaskA-GU-layerwiseattention.csv' 

final_list=[]
for filepath in all_files:
    all_tensors =[]
    predictions=[]
    create_list=[]
    spec =compute_features(filepath)
    spec=torch.from_numpy(np.ascontiguousarray(spec))
    all_tensors.append(spec)
    features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in all_tensors]))
    with torch.no_grad():
        features = features.to(device)
        preds,_,_,_ = model(features)
    predictions= preds.detach().cpu().numpy()>=0.5
    for item in predictions:
        if item:
            pred_label=1
        else:
            pred_label=0

    create_list.append(filepath.split('/')[-1])
    create_list.append(pred_label)
    final_list.append(create_list)
    print(' Label for {} is {}'.format(filepath,pred_label))


import csv
with open(csv_store, "w") as f:
    writer = csv.writer(f)
    writer.writerows(final_list)
    
    
    
