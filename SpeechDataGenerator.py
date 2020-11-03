#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
from utils import utils

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest):
        """
        Read the textfile and get the paths
        """
        self.json_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        
    def __len__(self):
        return len(self.json_links)
    
    
    def __getitem__(self, idx):
        json_link = self.json_links[idx]
        features,feat_lens,phonemes,phns_len,label = utils.load_data(json_link)
        
        sample = {'feats': torch.from_numpy(np.ascontiguousarray(features)), 
                  'feats_len': torch.from_numpy(np.ascontiguousarray(feat_lens)), 
                  'phonemes': torch.from_numpy(np.ascontiguousarray(phonemes)), 
                  'phns_len': torch.from_numpy(np.ascontiguousarray(phns_len)),
                  'label': torch.from_numpy(np.ascontiguousarray(label))}
        return sample
    
    
