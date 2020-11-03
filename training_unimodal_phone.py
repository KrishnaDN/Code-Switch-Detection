#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:45:06 2019
@author: krishna
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import yaml
import numpy as np
from torch import optim
import argparse
from sklearn.metrics import accuracy_score
from models.UniModal import PhonemeOnly
from utils.utils import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')




class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def train(model,dataloader_train,epoch,optimizer,device,criterion):
    total_loss=[]
    model.train()
    total_acc= []
    print('##################### Training######################')
    for i_batch, sample_batched in enumerate(dataloader_train):
        
        phonemes_ids = torch.stack(sample_batched[2]).long()
        phonemes_lens = torch.from_numpy(np.asarray(sample_batched[3]))
        labels  = torch.from_numpy(np.asarray(sample_batched[4]))
        
        phonemes_ids,phonemes_lens = phonemes_ids.to(device),phonemes_lens.to(device)   
        labels = labels.to(device)
        optimizer.zero_grad()
        prediction = model(phonemes_ids,phonemes_lens)
        loss = criterion(prediction,labels)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        pred_labels = np.argmax(prediction.detach().cpu().numpy(),axis=1)
        gt_labels = labels.detach().cpu().numpy()
        accuracy = accuracy_score(pred_labels,gt_labels)
        total_acc.append(accuracy)
        
        
    print('Training Loss {} after {} epochs'.format(np.mean(np.asarray(total_loss)),epoch))
    #print('Training CER {} after {} epochs'.format(avg_cer,epoch))
    print('Training Accuracy {} after {} epochs'.format(np.mean(np.asarray(total_acc)),epoch))
    return np.mean(np.asarray(total_loss)), np.mean(np.asarray(total_acc))
      
            
def evaluation(model,dataloader_test,epoch,device,criterion):
    model.eval()
    total_acc= []
    total_loss=[]
    print('##################### Training######################')
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            phonemes_ids = torch.stack(sample_batched[2]).long()
            phonemes_lens = torch.from_numpy(np.asarray(sample_batched[3]))
            labels  = torch.from_numpy(np.asarray(sample_batched[4]))
            
            phonemes_ids,phonemes_lens = phonemes_ids.to(device),phonemes_lens.to(device)   
            labels = labels.to(device)
            prediction = model(phonemes_ids,phonemes_lens)
            loss = criterion(prediction,labels)
            pred_labels = np.argmax(prediction.detach().cpu().numpy(),axis=1)
            gt_labels = labels.detach().cpu().numpy()
            accuracy = accuracy_score(pred_labels,gt_labels)
            total_acc.append(accuracy)
            total_loss.append(loss.item())
    
    #print('Training CER {} after {} epochs'.format(avg_cer,epoch))
    print('Testing Loss {} after {} epochs'.format(np.mean(np.asarray(total_loss)),epoch))
    print('Testing Accuracy {} after {} epochs'.format(np.mean(np.asarray(total_acc)),epoch))
    return np.mean(np.asarray(total_loss)), np.mean(np.asarray(total_acc))


def main(config):
    
    use_cuda = config['use_gpu']
    device = torch.device("cuda" if use_cuda==1 else "cpu")
    
    
    model = PhonemeOnly(config['vocab_len'],config['embedding_dim'],config['hidden_dim_PE'])
    model = model.to(device)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        config['n_warmup_steps'])
    criterion = nn.CrossEntropyLoss()    
    ### Data related
    dataset_train = SpeechDataGenerator(manifest=args.training_filepath)
    dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'],shuffle=True,collate_fn=speech_collate) 
    
    dataset_test = SpeechDataGenerator(manifest=args.testing_filepath)
    dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'] ,shuffle=True,collate_fn=speech_collate) 
    
    
    if not os.path.exists(args.save_modelpath):
        os.makedirs(args.save_modelpath)
    
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc=train(model,dataloader_train,epoch,optimizer,device,criterion)
        val_loss, val_acc = evaluation(model,dataloader_test,epoch,device,criterion)
        print('best accuracy so far {}'.format(best_acc))
        # Save
        if val_acc > best_acc: 
            best_acc = max(val_acc, best_acc)
            model_save_path = os.path.join(args.save_modelpath, 'best_check_point_'+str(epoch))
            state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
            torch.save(state_dict, model_save_path)
            
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

    
    
        
    

if __name__ == "__main__":
    ########## Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-training_filepath',type=str,default='meta/training_gujarati.txt')
    parser.add_argument('-testing_filepath',type=str, default='meta/testing_gujarati.txt')
    parser.add_argument('-config_file',type=str, default='config_gujarati.yaml')
    parser.add_argument('-save_modelpath',type=str, default='save_models_gujarati_phone_only/')
    
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    print(config)
    main(config)
    
