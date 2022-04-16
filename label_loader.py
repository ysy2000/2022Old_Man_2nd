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

#-*- coding: utf-8 -*-

import json
import csv
# 
def load_label_csv(label_path,max_length = 110):

    # Load file
    f = open(label_path,'r')
    sentence_list = csv.reader(f)

    txt2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    idx2txt = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
    max_length = max_length
    char_count = {}
    max_vocab_size = -1


    for sentence in sentence_list:
        for char in sentence:
            try:
                char_count[char] += 1
            except:
                char_count[char] = 1
    char_count = dict(sorted(char_count.items(), key=sort_target, reverse=True))

    txt2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    idx2txt = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
    if max_vocab_size == -1:
        for i, char in enumerate(list(char_count.keys())):
            txt2idx[char] = i + 4
            idx2txt[i + 4] = char
    else:
        for i, char in enumerate(list(char_count.keys())[:max_vocab_size]):
            txt2idx[char] = i + 4
            idx2txt[i + 4] = char

    return txt2idx, idx2txt 



# 데이콘 코드
def sort_target(x):
    return x[1]
    
def load_label_Dacon(max_length, max_vocab_size=-1):
    txt2idx = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
    idx2txt = {0:'<pad>', 1:'<unk>', 2:'<sos>', 3:'<eos>'}
    max_length = max_length
    char_count = {}
    max_vocab_size = max_vocab_size
    
    # train 입력 리스트
    for sentence in tqdm(sentence_list):
        for char in sentence:
            try:
                # 이전에 이미 진행한 글자.
                char_count[char] += 1

            except:
                # 처음 보는 글자
                char_count[char] = 1
    char_count = dict(sorted(char_count.items(), key=sort_target, reverse=True))
    
    txt2idx = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
    idx2txt = {0:'<pad>', 1:'<unk>', 2:'<sos>', 3:'<eos>'}

    if max_vocab_size == -1: #vocab 사이즈 전체 
        for i, char in enumerate(list(char_count.keys())):
            txt2idx[char]=i+4
            idx2txt[i+4]=char
    else:
        for i, char in enumerate(list(char_count.keys())[:max_vocab_size]):
            txt2idx[char]=i+4
            idx2txt[i+4]=char

    return txt2idx, idx2txt
        



# main에서 쓰는 함수 - json 사용
def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()     # [char] = id
        index2char = dict()     # [id] = ch

        # json 일렬로 선 id가 
        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char
            
        return char2index, index2char

# index, char, freq 순으로 표시된 txt 파일 
def load_label_index(label_path):
    char2index = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    index2char = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}

    #print(label_path)
    with open(label_path, 'r', encoding="utf-8") as f:
        for no, line in enumerate(f):
            if line[0:2] == 'id': 
                #print("skip label")
                continue

            index, char, freq = line.strip().split(',')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index) +4
            index2char[int(index)+4] = char

    return char2index, index2char


    wav_paths = list() 
    script_paths = list()
    
