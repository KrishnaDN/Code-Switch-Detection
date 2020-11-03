#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 01:00:28 2020

@author: krishna
"""

from models.Audio_Encoder import Audio_Encoder
from models.Phoneme_Encoder import Phoneme_Encoder
import torch
import torch.nn.functional as F
import torch.nn as nn

class AudioOnly(nn.Module):
    def __init__(self,input_dim,hidden_dim_AE, n_layers=1,dropout=0.3):
        super(AudioOnly, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_AE  = hidden_dim_AE
        self.n_layers = n_layers
        
        self.audio_encoder = Audio_Encoder(self.input_dim, self.hidden_dim_AE,self.n_layers)
        self.linear = nn.Linear(4*hidden_dim_AE,512)
        self.proj = nn.Linear(512,128)
        self.output = nn.Linear(128,2)
        
        
        
    def forward(self,input_feats, input_lens):
        audio_enc_out = self.audio_encoder(input_feats,input_lens)
        ### Stat pool audio encoder
        mean = torch.mean(audio_enc_out,dim=2)
        var  = torch.var(audio_enc_out,dim=2)
        stat_pool_audio = torch.cat((mean,var),dim=1)
    
        
        ### Combine networks
        concat_rep = stat_pool_audio
        linear = F.relu(self.linear(concat_rep))
        proj = F.relu(self.proj(linear))
        output = self.output(proj)
        return output

class PhonemeOnly(nn.Module):
    def __init__(self,vocab_len, embedding_dim,hidden_dim_PE, n_layers=1,dropout=0.3):
        super(PhonemeOnly, self).__init__()
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.hidden_dim_PE =hidden_dim_PE
        self.n_layers = n_layers
        
        self.phoneme_encoder = Phoneme_Encoder(self.vocab_len, self.embedding_dim,self.hidden_dim_PE,self.n_layers)
        self.linear = nn.Linear(4*hidden_dim_PE,512)
        self.proj = nn.Linear(512,128)
        self.output = nn.Linear(128,2)
        
        
        
    def forward(self,phonemes, phoneme_lens):
        
        phoneme_enc_out = self.phoneme_encoder(phonemes,phoneme_lens)
    
        ### Stat pool phoneme encoder
        mean = torch.mean(phoneme_enc_out,dim=2)
        var  = torch.var(phoneme_enc_out,dim=2)
        stat_pool_phoneme = torch.cat((mean,var),dim=1)
        
        ### Combine networks
        concat_rep = stat_pool_phoneme
        linear = F.relu(self.linear(concat_rep))
        proj = F.relu(self.proj(linear))
        output = self.output(proj)
        return output




