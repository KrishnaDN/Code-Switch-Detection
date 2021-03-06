#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:00:22 2020

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
import torch.nn as nn
from modules.transformer import TransformerEncoder



class Convolution_Block(nn.Module):
    def __init__(self, input_dim=13,cnn_out_channels=64):
        super(Convolution_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out_channels, kernel_size=7, stride=3,padding=3),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels,cnn_out_channels, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )
        
    def forward(self, inputs):
        out = self.conv(inputs)
        return out

class Transformer(nn.Module):
    def __init__(self, embed_dim,num_heads=4,layers=1):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.attn_dropout = 0.1
        self.relu_dropout=0.1
        self.res_dropout = 0.1
        self.embed_dropout=0.1
        self.attn_mask = False
        self.layers = layers
        
        self.transformer = TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=self.layers,
                                  attn_dropout=self.attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
        
    def forward(self,inputs_q,inputs_k):
        x_q = inputs_q.permute(1,0,2)
        x_k = inputs_k.permute(1,0,2)
        x_v = x_k
        transformer_out = self.transformer(x_q,x_k,x_v)
        transformer_out = transformer_out.permute(1,2,0)
        return transformer_out



class Audio_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2,dropout=0.3,cnn_out_channels=64,rnn_celltype='gru'):
        super(Audio_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_p = dropout
      
        self.conv = Convolution_Block(self.input_dim,cnn_out_channels=cnn_out_channels)
        
        if rnn_celltype == 'lstm':
            self.rnn =  nn.LSTM(cnn_out_channels, self.hidden_dim, self.n_layers, dropout=self.dropout_p, bidirectional=True,batch_first=True)
        else:
            self.rnn =  nn.GRU(cnn_out_channels, self.hidden_dim, self.n_layers, dropout=self.dropout_p, bidirectional=True,batch_first=True)
        
        self.transformer = Transformer(embed_dim=2*hidden_dim,num_heads=4)

    def forward(self, inputs, input_lengths):
        
        output_lengths = self.get_conv_out_lens(input_lengths)
        out = self.conv(inputs) 
        out = out.permute(0,2,1)
        
        out = nn.utils.rnn.pack_padded_sequence(out, output_lengths, enforce_sorted=False, batch_first=True)
        out, rnn_hidden_state = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        rnn_out = out.transpose(0, 1)
        transformer_out = self.transformer(rnn_out,rnn_out)
        return transformer_out



    def get_conv_out_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv1d :
                seq_len = ((seq_len + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) / m.stride[0] + 1)

        return seq_len.int()
