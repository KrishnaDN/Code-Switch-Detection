#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:50:04 2020

@author: krishna
"""



import os
import numpy as np
import json
import torch
import librosa

def load_data(json_link):
    with open(json_link) as json_file:
        data = json.load(json_file)
    
    npy_filepath = data['audio_filepath']
    load_npy = np.load(npy_filepath,allow_pickle=True).item()
    features = load_npy['feats']
    feat_lens = load_npy['len']
    phonemes = data['phn_ids']
    phns_len = data['phn_lens']
    label = data['label']
    return features,feat_lens,phonemes,phns_len,label


def feature_stack(feature):
    feat_dim,time=feature.shape
    stacked_feats=[]
    for i in range(0,time-3,3):
        splice = feature[:,i:i+3]
        stacked_feats.append(np.array(splice).flatten())
    return np.asarray(stacked_feats)


def SpecAugment(stacked_feature):
    time,feat_dim=stacked_feature.shape
    ##### Masking 5% of the data
    win_len = round(time*0.05)
    mask_start_index = np.random.randint(0, time-win_len)
    create_zero_mat = np.zeros((win_len,feat_dim))
    stacked_feature[mask_start_index:mask_start_index+win_len,:] = create_zero_mat
    masked_features = stacked_feature
    return masked_features


def pad_labels(labels,pad_token,max_len):
    input_len=len(labels)
    if input_len<max_len:    
        pad_len=max_len-input_len
        pad_seq = torch.fill_(torch.zeros(pad_len), pad_token).long()
        labels = torch.cat((labels,pad_seq))
    return labels



def pad_sequence_feats(features_list,feat_lengths):
    lengths =feat_lengths
    max_length = max(lengths)
    padded_feat_batch=[]
    for feature_mat in features_list:
        pad_mat = torch.zeros((max_length-feature_mat.shape[0],feature_mat.shape[1]))
        padded_feature = torch.cat((feature_mat,pad_mat),0)
        padded_feat_batch.append(padded_feature.T)
    return padded_feat_batch



def pad_sequence_phonemes(phonemes,phn_lengths,PAD_TOKEN=6):
    lengths =phn_lengths
    max_length = max(lengths)
    padded_phonemes_batch=[]
    for phoneme in phonemes:
        pad_mat = torch.fill_(torch.zeros((max_length-phoneme.shape[0])), PAD_TOKEN).long()
        padded_phonemes = torch.cat((phoneme,pad_mat),0)
        padded_phonemes_batch.append(padded_phonemes)
    return padded_phonemes_batch


def speech_collate(batch):
    features=[]
    feat_lengths=[]
    phonemes = []
    phn_lengths = []
    labels = []
    for item in batch:
        features.append(item['feats'])
        feat_lengths.append(int(item['feats_len']))
        phonemes.append(item['phonemes'])
        phn_lengths.append(int(item['phns_len']))
        labels.append(int(item['label']))
         
    padded_feats = pad_sequence_feats(features,feat_lengths)
    padded_phns = pad_sequence_phonemes(phonemes,phn_lengths,PAD_TOKEN=6)

    return padded_feats,feat_lengths, padded_phns,phn_lengths,labels

def create_vocab_dict(vocab_path):
    vocab_list = [line.rstrip('\n') for line in open(vocab_path)]
    i=0
    vocab_dict={}
    for item in vocab_list:
        vocab_dict[item] = i
        i+=1
    return vocab_dict


