
#-*- coding: utf-8 -*-

"""
Copyright 2019-present NAVER Corp.
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

import time

import os
import json
import math
import random
import argparse
import numpy as np
#from tqdm import tqdm
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim

import Levenshtein as Lev 

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

import label_loader
from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler

from models import EncoderRNN, DecoderRNN, Seq2Seq


char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length 


def get_distance(ref_labels, hyp_labels):
    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        transcripts.append('{ref}\n>>{hyp}'.format(hyp=hyp, ref=ref))

        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 

    return total_dist, total_length, transcripts



def evaluate(train_name,model, data_loader, criterion, device, save_output=False):
    total_loss = 0.
    #total_num = 0
    total_dist = 0
    total_length = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad():
        #for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader)):
        for i, (data) in enumerate(data_loader):
            feats, scripts, feat_lengths, script_lengths = data

            feats = feats.to(device)
            scripts = scripts.to(device)
            feat_lengths = feat_lengths.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            #logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            dist, length, transcripts = get_distance(target, y_hat)
            cer = float(dist / length) * 100

            total_dist += dist
            total_length += length
            if save_output :
                #transcripts_list += transcripts 
                ##################LOG######################################################
                for transcript in transcripts:
                    with open("/home/jyseo/DNN_zeroth/log/train_log/"+train_name+"_TestRESULT.txt", "a") as f:
                        f.write(transcript+'\n')

    aver_cer = float(total_dist / total_length) * 100
    return  aver_cer

def inference(model, data_loader, criterion, device, save_output=True):

    inf_list = []

    model.eval()
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            feats, scripts, feat_lengths, script_lengths = data

            feats = feats.to(device)
            scripts = scripts.to(device)
            feat_lengths = feat_lengths.to(device)

            src_len = scripts.size(1)
            #target = scripts[:, 1:]

            logit = model(feats, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            for i in range(len(y_hat)):
                inf = label_to_string(y_hat[i])
                inf_list += inf

    return inf_list


def main():
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token



    parser = argparse.ArgumentParser(description='LAS')
    parser.add_argument('--model-name', type=str, default='LAS')

    ## 실험 마다 이름 설정 필요
    parser.add_argument('--train_name', type=str, default='new_zeroth_KRAoudio_SA_TF9_BEST')
    parser.add_argument('--model-path', default='/home/jyseo/DNN_zeroth/saved_models/new_zeroth_KRAoudio_SA_TF9_Best.pth', help='model to load')  #  불러올 모델 path 

    # Dataset
    # data file PATH 
    parser.add_argument('--test-file-list', nargs='*',
                        help='data list about test dataset', default='/home/jyseo/DNN_zeroth/data/zeroth_test_list.csv')


    parser.add_argument('--labels-path', default='/home/jyseo/DNN_zeroth/data/labelForChars.csv', help='Contains large characters over korean')# 음절 리스트 for token
    parser.add_argument('--dataset-path', default='/home/jyseo/DNN_zeroth/data', help='Target dataset path')

    # Hyperparameters
    parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
    parser.add_argument('--encoder_layers', type=int, default=3, help='number of layers of model (default: 3)')
    parser.add_argument('--encoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
    parser.add_argument('--decoder_size', type=int, default=512, help='hidden size of model (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True, help='Turn off bi-directional RNNs, introduces lookahead convolution')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size in training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of gpus (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--learning-anneal', default=1, type=float, help='Annealing learning rate every epoch')
    parser.add_argument('--teacher_forcing', type=float, default=0.9, help='Teacher forcing ratio in decoder (default: 1.0)')
    parser.add_argument('--max_len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    # Audio Config
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
    parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
    parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')
    # System    
    ## 경로 설정 필요 
    parser.add_argument('--save-folder', default='/home/jyseo/DNN_zeroth/saved_models', help='Location to save epoch models')

    #parser.add_argument('--log-path', default='/home/jyseo/AI2021KR/log/', help='path to predict log about valid and test dataset')
    
    #parser.add_argument('--cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123456, help='random seed (default: 123456)')
    parser.add_argument('--mode', type=str, default='train', help='Train or Test')
    parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                        help='Finetune the model after load model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    char2index, index2char = label_loader.load_label_index(args.labels_path)
    #print(char2index)
    SOS_token = char2index['<sos>']
    EOS_token = char2index['<eos>']
    PAD_token = char2index['<pad>']

    for i in range(7):
        print("<index2char[",i,"]:", index2char[i],">")

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    args.num_gpu = torch.cuda.device_count()
    print("using ",args.num_gpu,"GPUs")

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride)

    # Batch Size
    batch_size = args.batch_size #* args.num_gpu

    print(">> test dataset at: ", args.test_file_list) # parse에서 받는 것은 csv 위치 
    val_Data_list = pd.read_csv(args.test_file_list) 

    if args.num_gpu != 1:

        last_batch = len(val_Data_list) % batch_size
        if last_batch != 0 and last_batch < args.num_gpu:
            # 배치에 안맞는 마지막 데이터 아예 빼버림
            print("out ",last_batch, " batches")
            val_Data_list = val_Data_list[:-last_batch]

    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                    dataset_path=args.dataset_path,  # train, tset 데이터 따로 있음
                                    data_list=val_Data_list,
                                    char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                    normalize=True,
                                    SAflag = False)
    test_Loader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)


    input_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1)
    enc = EncoderRNN(input_size, args.encoder_size, n_layers=args.encoder_layers,
                     dropout_p=args.dropout, bidirectional=args.bidirectional, 
                     rnn_cell=args.rnn_type, variable_lengths=False)

    dec = DecoderRNN(len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                     SOS_token, EOS_token,
                     n_layers=args.decoder_layers, rnn_cell=args.rnn_type, 
                     dropout_p=args.dropout, bidirectional_encoder=args.bidirectional)


    model = Seq2Seq(enc, dec)

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    optim_state = None

    print("Loading checkpoint model %s" % args.model_path)
    state = torch.load(args.model_path)
    model.load_state_dict(state['model'])
    print('Model loaded')

 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)\

    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    if optim_state is not None:
            optimizer.load_state_dict(optim_state)

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=PAD_token).to(device)

    train_model = nn.DataParallel(model)

    train_start = time.time()


    # Test
    test_cer = evaluate(args.train_name,model, test_Loader, criterion, device, save_output=True)
    test_log = '\n>>Test CER {cer:.3f}\t\n'.format(cer=test_cer)
    print(test_log)

    # Log Summary
    with open("/home/jyseo/DNN_zeroth/log/train_log/"+args.train_name+"_TestRESULT.txt", "a") as ff:
        ff.write(test_log + "\n")

if __name__ == "__main__":
    main()
