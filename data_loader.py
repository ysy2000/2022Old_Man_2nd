import os
import math
import torch
import librosa
import numpy as np
import scipy.signal

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from specAug.spec_augment import spec_augment

import matplotlib.pyplot as plt 

def load_audio(path):

    # 하드 디스크에 저장. ssd에 저장할거면 numpy
    sound = np.memmap(path, dtype='h', mode='r')
    sound = sound.astype('float32') / 32767

    # 비었으면 경고
    assert len(sound)

    # 한칸짜리로 변환 : 원래형태는 어떻지?? 
    sound = torch.from_numpy(sound).view(-1, 1).type(torch.FloatTensor) # torch.Tenor를 사용하면 사본이라 원본은 변환 안됨
    sound = sound.numpy() 

    # 
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average

    return sound

#3
class SpectrogramDataset(Dataset):
    def __init__(self, audio_conf, dataset_path, data_list, char2index, sos_id, eos_id, normalize=False, SAflag = False):
        super(SpectrogramDataset, self).__init__()
        """
        Dataset loads data from a list contatining wav_name, transcripts, speaker_id by dictionary.
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds.
        :param data_list: List of dictionary. key : "wav", "text", "speaker_id"
        :param char2index: Dictionary mapping character to index value.
        :param sos_id: Start token index.
        :param eos_id: End token index.
        :param normalize: Normalized by instance-wise standardazation.
        """
        self.audio_conf = audio_conf
        self.data_list = data_list
        self.size = len(self.data_list)
        self.char2index = char2index
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.PAD = 0
        self.normalize = normalize
        self.dataset_path = dataset_path
        self.SAflag = SAflag

    def __getitem__(self, index):
        #wav_name = self.data_list[index]['wav']    #json
        wav_name = self.data_list['file_name'][index]
        audio_path = self.dataset_path+ wav_name
        
        #transcript = self.data_list[index]['text'] #json
        transcript = self.data_list['text'][index]
        #print(self.dataset_path,wav_name)
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript)



        if self.SAflag :
            self.save_test_img(index, spect, "spech_Before" ,basePoint = 360)
            #print(index)
            ##########################
            # SpecAugment : 전체
            time_warping_para= 0#40    # 80 너무 김
            frequency_masking_para=15 #27
            time_masking_para= 45 #100
            frequency_mask_num= 3 #1    
            time_mask_num= 3 #1
            
            ########################
            #print(type(spect))
            spect = spec_augment(spect, time_warping_para= time_warping_para, frequency_masking_para=frequency_masking_para,
                                            time_masking_para=time_masking_para, frequency_mask_num=frequency_mask_num, time_mask_num=time_mask_num)
            
            self.save_test_img(index, spect, "spech_After" ,basePoint = 360)
   
        """
        if spect.dim() == 1:
            with open("/home/jyseo/Hackathon2021/data/out.txt", "a") as f:
                f.write(wav_name)
                f.write(self.data_list['text'][index])
            print("차원 오류",wav_name,self.data_list['text'][index])

        #try: print(len(spect),spect.size(1))
        #except : print("차원 오류",wav_name,self.data_list['text'][index])
        """
        return spect, transcript


    def save_test_img(self,index, tensor_image, Name ,basePoint = 360):

        # 시험 저장
        #print(index)
        if index == basePoint: 
            audio_features = tensor_image.numpy()
            librosa.display.specshow(audio_features, sr= 16000 , x_axis='time', y_axis=None)
            plt.imshow
            plt.title(Name)
            plt.tight_layout()
            plt.savefig("/home/ysy/2021AI_data_Hackarthon/log/"+Name+"_specIMG.png") 
            print(Name + " test img saved")

    def parse_audio(self, audio_path):
        y = load_audio(audio_path)

        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
        window_size = n_fft
        stride_size = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])

        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=stride_size, win_length=window_size, window=scipy.signal.hamming)
        spect, phase = librosa.magphase(D)

        """#Mel 추가
        #####################
        sr = 22050
        n_mels = 64    
        #######################
        mel_spec = librosa.feature.melspectrogram(S=spect, sr=sr, n_mels= n_mels, hop_length=stride_size, win_length=window_size)
        feature = librosa.amplitude_to_db(mel_spec, ref=0.00002)
        """
        # S = log(S+1)
        #spect = np.log1p(spect)
        if self.normalize:
            mean = np.mean(spect)
            std = np.std(spect)
            spect -= mean
            spect /= std

        spect = torch.FloatTensor(spect)

        return spect

    def parse_transcript(self, transcript):
        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        transcript = [self.sos_id] + transcript + [self.eos_id]
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):

    def seq_length_(p):
        return p[0].size(1)
    def target_length_(p):
        return len(p[1])


    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)


    seq_lengths    = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_size = max(seq_lengths)
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths


#1
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

#2
class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
