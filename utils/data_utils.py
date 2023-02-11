import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F  
import random
from utils.utils import frac_idx


class VideoFeatureDict():
    def __init__(self, root, max_vlen, debug):
        self.debug = debug
        self.path_dict = dict()
        self.video_features = dict()
        filenames = glob.glob(os.path.join(root, "*.npy"))
        self.max_vlen = max_vlen
        # self.sample_method = sample_method

        if debug:
            for filename in tqdm(filenames, total=len(filenames), desc="load video path"):
                video_id = filename.split("/")[-1].split(".")[0]
                self.path_dict[video_id] = filename
        else:
            for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
                video_id = filename.split("/")[-1].split(".")[0]
                feature = np.load(filename)
                feature = torch.FloatTensor(feature)
                self.video_features[video_id] = feature

    def __getitem__(self, k):
        if self.debug:
            filename = self.path_dict[k]
            feature = np.load(filename)
            feature = torch.FloatTensor(feature)
            return feature
        else:
            return self.video_features[k]

def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_seq(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_seq(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_seq(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_seq(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_video_seq(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = torch.zeros(size=[add_length, feature_length], dtype=seq.dtype)
            seq_ = torch.cat([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length



#  ---- video augmentation, then sampling ---

def select_negtive_segment(seglen, vfeat, label):
    neg_vfeat = vfeat[torch.where(label==0)[0]]

    if neg_vfeat.shape[0] == 0:
        neg_vfeat = torch.rand_like(vfeat)

    while len(neg_vfeat) < seglen:
        neg_vfeat = torch.cat([neg_vfeat, neg_vfeat])
    r = random.randint(0, len(neg_vfeat)-seglen)
    seg_vfeat = neg_vfeat[r:r+seglen,:]
    return seg_vfeat
    
def label_idx(label, threshold=0.01):
    sidx = min(torch.where(label>=threshold)[0]).item()
    eidx = max(torch.where(label>=threshold)[0]).item()
    return sidx, eidx

def feature_dilation(vfeat, label, p):
    vlen = vfeat.shape[0]
    head_len = int(round(random.random() * p * vlen))
    tail_len = int(round(random.random() * p * vlen))
    
    head_label = torch.zeros(head_len)
    tail_label = torch.zeros(tail_len)

    head_vfeat = select_negtive_segment(head_len, vfeat, label)
    tail_vfeat = select_negtive_segment(tail_len, vfeat, label)

    new_vfeat = torch.cat([head_vfeat, vfeat, tail_vfeat])
    new_label = torch.cat([head_label, label, tail_label])
    
    return new_vfeat, new_label


def feature_erosion(vfeat, label, p):
    ori_sidx, ori_eidx = label_idx(label)
    vlen = vfeat.shape[0]
    while True:
        head_len = int(round(random.random() * p * vlen))
        if 0 <= head_len <= ori_sidx:
            break

    while True:
        tail_len = vlen - 1 - int(round(random.random() * p * vlen))
        if ori_eidx <= tail_len <= vlen-1:
            break
    new_vfeat = vfeat[head_len:tail_len+1]
    new_label = label[head_len:tail_len+1]
    return new_vfeat, new_label

def video_augmentation(sfrac, efrac, vfeat, aug):
    vlen = vfeat.shape[0]
    label = torch.zeros(vlen)
    sidx, eidx = frac_idx([sfrac, efrac], vlen)
    label[sidx:eidx+1] = 1
    # print(label)
    k = random.choice(list(aug.keys()))
    if k == "unchanged":
        new_vfeat, new_label = vfeat, label
    elif k == "dilation":
        new_vfeat, new_label = feature_dilation(vfeat, label, aug[k])
    elif k == "erosion":
        new_vfeat, new_label = feature_erosion(vfeat, label, aug[k])
    else:
        raise NotImplemented
    assert new_vfeat.shape[0] == new_label.shape[0]
    # print(new_label)

    return new_vfeat, new_label


def interpolate_avrage(x, size):
    vlen = x.shape[0]
    idxs = torch.arange(0, size, 1.0) / size * (vlen-1) 
    idxs = torch.cat([idxs, torch.tensor([vlen])])
    idxs = torch.round(idxs).int()
    new_x = []
    for i in range(size):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_x.append(torch.mean(x[s_idx:e_idx], axis=0))
        else:
            new_x.append(x[s_idx])
    new_x = torch.stack(new_x)
    return new_x

def sample_vfeat_linear(vfeat, label, max_vlen, sample_method):
    if sample_method == "original":
        new_vfeat = vfeat
        new_label = label

    elif sample_method == "truncation":
        vlen = vfeat.shape[0]
        if vlen <= max_vlen:
            new_vfeat, new_label = vfeat, label
        else:
            new_vfeat = interpolate_avrage(vfeat, max_vlen)
            new_label = interpolate_avrage(label, max_vlen)

    elif sample_method == "samelen":
        new_vfeat = interpolate_avrage(vfeat, max_vlen)
        new_label = interpolate_avrage(label, max_vlen)

        # new_vfeat = torch.nn.functional.interpolate(vfeat.transpose(0, 1).unsqueeze(0), size=max_vlen, mode='area')
        # new_vfeat = new_vfeat[0, ...].transpose(0, 1)

        # new_label = torch.nn.functional.interpolate(label.transpose(0, 1).unsqueeze(0), size=max_vlen, mode='area')
        # new_label = new_label[0, ...].transpose(0, 1)
    else:
        raise
    return new_vfeat, new_label