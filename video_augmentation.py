import numpy as np
import json
from utils.utils import time_idx, frac_idx
import random
import torch

random.seed(3)

def select_negtive_segment(seglen, vfeat, label):
    neg_vfeat = vfeat[torch.where(label==0)[0]]

    while len(neg_vfeat) < seglen:
        neg_vfeat = torch.cat([neg_vfeat, neg_vfeat])
    r = random.randint(0, len(neg_vfeat)-seglen)
    seg_vfeat = neg_vfeat[r:r+seglen,:]
    return seg_vfeat
    

def feature_lengthening(vfeat, label, head_len, tail_len):

    head_label = torch.zeros(head_len)
    tail_label = torch.zeros(tail_len)

    head_vfeat = select_negtive_segment(head_len, vfeat, label)
    tail_vfeat = select_negtive_segment(tail_len, vfeat, label)

    new_vfeat = torch.cat([head_vfeat, vfeat, tail_vfeat])
    new_label = torch.cat([head_label, label, tail_label])
    
    return new_vfeat, new_label

def label_idx(label, threshold=0.01):
    sidx = min(torch.where(label>=threshold)[0])
    eidx = max(torch.where(label>=threshold)[0])
    return sidx, eidx


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

def video_augmentation(sfrac, efrac, vfeat, aug_type):
    vlen = vfeat.shape[0]
    label = torch.zeros(vlen)
    sidx, eidx = frac_idx([sfrac, efrac], vlen)
    label[sidx:eidx+1] = 1
    print(label)
    alpha = 0.1
    if aug_type == "lengthening":
        head_len = int(round(random.random() * alpha * vlen))
        tail_len = int(round(random.random() * alpha * vlen))
        new_vfeat, new_label = feature_lengthening(vfeat, label, head_len, tail_len)
    
    elif aug_type == "erosion":
        new_vfeat, new_label = feature_erosion(vfeat, label, alpha)
        return new_vfeat, new_label


def sample_vfeat_linear(vfeat, label, max_vlen, sample_method):
    if sample_method == "original":
        new_vfeat = vfeat
        new_label = label

    elif sample_method == "truncation":
        vlen = vfeat.shape[0]
        if vlen <= max_vlen:
            new_vfeat = vfeat
            new_label = label
        else:
            idxs = torch.arange(0, max_vlen + 1, 1.0) / max_vlen * vlen
            idxs = torch.round(idxs).int()
            idxs[idxs > vlen - 1] = vlen - 1
            new_vfeat = []
            new_label = []
            for i in range(max_vlen):
                s_idx, e_idx = idxs[i], idxs[i + 1]
                if s_idx < e_idx:
                    new_vfeat.append(torch.mean(vfeat[s_idx:e_idx], axis=0))
                    new_label.append(torch.mean(label[s_idx:e_idx], axis=0))
                else:
                    new_vfeat.append(vfeat[s_idx])
                    new_label.append(label[s_idx])
            new_vfeat = torch.stack(new_vfeat)
            new_label = torch.stack(new_label)

    elif sample_method == "samelen":
        new_vfeat = torch.nn.functional.interpolate(vfeat.transpose(0, 1).unsqueeze(0), 
                        size=max_vlen, mode='linear', align_corners=False)
        new_vfeat = new_vfeat[0, ...].transpose(0, 1)

        new_label = torch.nn.functional.interpolate(label.transpose(0, 1).unsqueeze(0), 
                        size=max_vlen, mode='linear', align_corners=False)
        new_label = new_label[0, ...].transpose(0, 1)
    else:
        raise
    return new_vfeat, new_label

sfrac, efrac = 0.02, 0.98
vfeat_path = "/storage_fast/rjliang/activitynet/i3d/v_KgfKmcsEMK0.npy"
vfeat = torch.from_numpy(np.load(vfeat_path))

vfeat, label = video_augmentation(sfrac, efrac, vfeat, aug_type="erosion")
vfeat, label = sample_vfeat_linear(vfeat, label, max_vlen=100, sample_method="truncation")
sidx, eidx = label_idx(label, threshold=0.1)
print(label)
print(sidx, eidx)


