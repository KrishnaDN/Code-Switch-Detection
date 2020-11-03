#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:22:15 2020

@author: krishna
"""

import os
import numpy as np
import glob
import soundfile as sf
#### Task A





label_dict={'G':1,'T':1,'E':0}
def compute_dur(file_list):
    dur_list =[]
    for filepath in file_list:
        audio_data,fs = sf.read(filepath)
        dur = len(audio_data)/float(fs)
        dur_list.append(dur)
    return dur_list


def create_dict(read_csv):
    create_dict ={}
    for row in read_csv:
        filename = row.split('\t')[0]
        labels = row.split('\t')[1]
        labels = ''.join(labels.split(' '))
        if all(x == labels[0] for x in labels):
            cor_label = 1
        else:
            cor_label=0
        create_dict[filename]=cor_label
    return create_dict



def create_dict_CTC(read_csv):
    create_dict ={}
    for row in read_csv:
        filename = row.split('\t')[0]
        labels = row.split('\t')[1]
        labels = ''.join(labels.split(' '))
        #if all(x == labels[0] for x in labels):
        #    cor_label = 1
        #else:
        #    cor_label=0
        save_list=[labels,len(labels)]
        create_dict[filename]=save_list
    return create_dict


if __name__=='__main__':
    root_path = '/media/newhd/MS_LID_challenge/dataset_task_A'
    all_langs_folders = sorted(glob.glob(root_path+'/*/'))
    for lang_folder in all_langs_folders:
        train_path=lang_folder+'Train/'
        dev_path = lang_folder+'Dev/'
        all_train_files = sorted(glob.glob(train_path+'/Audio/*.wav'))
        all_dev_files = sorted(glob.glob(dev_path+'/Audio/*.wav'))
        
        #dur_list = compute_dur(all_train_files)
        csv_files = sorted(glob.glob(train_path+'*.tsv'))
        if not csv_files[0]:
            print('CSV file not found')
            break
        csv_filepath = csv_files[0]
        read_csv = [line.rstrip('\n') for line in open(csv_filepath)]
        ######
        train_dict = create_dict_CTC(read_csv)
        
        csv_files = sorted(glob.glob(dev_path+'*.tsv'))
        if not csv_files[0]:
            print('CSV file not found')
            break
        csv_filepath = csv_files[0]
        read_csv = [line.rstrip('\n') for line in open(csv_filepath)]
        test_dict = create_dict_CTC(read_csv)
        
        
        ######################
        training_list_path = 'training_'+lang_folder.split('/')[-2].split('_')[1]+'.txt'
        testing_list_path = 'testing_'+lang_folder.split('/')[-2].split('_')[1]+'.txt'
        
        create_train_fid = open('meta/'+training_list_path,'w')
        for filepath in all_train_files:
            npy_filepath=filepath[:-4]+'.npy'
            label = train_dict[filepath.split('/')[-1][:-4]]
            store_label=[]
            store=[]
            for item in label[0]:
                store_label.append(label_dict[item])
            store.append(store_label)
            store.append(label[1])
            np.save(npy_filepath,store)
            
            to_write = filepath
            create_train_fid.write(to_write+'\n')
        create_train_fid.close()
        
        create_test_fid = open('meta/'+testing_list_path,'w')
        for filepath in all_dev_files:
            npy_filepath=filepath[:-4]+'.npy'
            label = test_dict[filepath.split('/')[-1][:-4]]
            store_label=[]
            store=[]
            for item in label[0]:
                store_label.append(label_dict[item])
            store.append(store_label)
            store.append(label[1])
            np.save(npy_filepath,store)
            
            to_write = filepath
            create_test_fid.write(to_write+'\n')
        create_test_fid.close()
    
#############################################################################



















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

