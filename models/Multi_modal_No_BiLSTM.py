#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 01:00:28 2020

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

from models.Audio_Encoder_No_BiLSTM import Audio_Encoder
from models.Phoneme_Encoder_No_BiLSTM import Phoneme_Encoder
import torch
import torch.nn.functional as F
import torch.nn as nn

class Multi_modal(nn.Module):
    def __init__(self,input_dim, vocab_len, embedding_dim):
        super(Multi_modal, self).__init__()
        self.input_dim = input_dim
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        
        
        self.audio_encoder = Audio_Encoder(self.input_dim)
        self.phoneme_encoder = Phoneme_Encoder(self.vocab_len, self.embedding_dim)
        self.linear = nn.Linear(256,512)
        self.proj = nn.Linear(512,128)
        self.output = nn.Linear(128,2)
        
        
        
    def forward(self,input_feats, input_lens,phonemes, phoneme_lens):
        audio_enc_out = self.audio_encoder(input_feats)
        phoneme_enc_out = self.phoneme_encoder(phonemes)
        ### Stat pool audio encoder
        mean = torch.mean(audio_enc_out,dim=2)
        var  = torch.var(audio_enc_out,dim=2)
        stat_pool_audio = torch.cat((mean,var),dim=1)
        
        ### Stat pool phoneme encoder
        mean = torch.mean(phoneme_enc_out,dim=2)
        var  = torch.var(phoneme_enc_out,dim=2)
        stat_pool_phoneme = torch.cat((mean,var),dim=1)
        
        ### Combine networks
        concat_rep = torch.cat((stat_pool_audio,stat_pool_phoneme),dim=1)
        linear = F.relu(self.linear(concat_rep))
        proj = F.relu(self.proj(linear))
        output = self.output(proj)
        return output





