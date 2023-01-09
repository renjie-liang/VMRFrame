import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F  

# def visual_feature_sampling(visual_feature, max_num_clips): ### ??? using cv2 nearst
#     num_clips = visual_feature.shape[0]
#     if num_clips <= max_num_clips:
#         return visual_feature
#     idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
#     idxs = np.round(idxs).astype(np.int32)
#     idxs[idxs > num_clips - 1] = num_clips - 1
#     new_visual_feature = []
#     for i in range(max_num_clips):
#         s_idx, e_idx = idxs[i], idxs[i + 1]
#         if s_idx < e_idx:
#             new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
#         else:
#             new_visual_feature.append(visual_feature[s_idx])
#     new_visual_feature = np.asarray(new_visual_feature)
#     return new_visual_feature


# def load_video_features(root, max_vlen):
#     video_features = dict()
#     filenames = glob.glob(os.path.join(root, "*.npy"))
#     for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
#         video_id = filename.split("/")[-1].split(".")[0]
#         feature = np.load(filename)
#         feature = torch.FloatTensor(feature)
#         video_features[video_id] = sample_vfeat_linear(feature, max_vlen)
#     return video_features
class VideoFeatureDict():
    def __init__(self, root, max_vlen, debug):
        self.debug = debug
        self.max_vlen = max_vlen
        self.path_dict = dict()
        self.video_features = dict()

        filenames = glob.glob(os.path.join(root, "*.npy"))
        if debug:
            for filename in tqdm(filenames, total=len(filenames), desc="load video path"):
                video_id = filename.split("/")[-1].split(".")[0]
                self.path_dict[video_id] = filename
        else:
            for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
                video_id = filename.split("/")[-1].split(".")[0]
                feature = np.load(filename)
                feature = torch.FloatTensor(feature)
                self.video_features[video_id] = sample_vfeat_linear(feature, self.max_vlen)

    def __getitem__(self, k):
        if self.debug:
            filename = self.path_dict[k]
            feature = np.load(filename)
            feature = torch.FloatTensor(feature)
            feature = sample_vfeat_linear(feature, self.max_vlen)
            return feature
        else:
            return self.video_features[k]


def sample_vfeat_linear(v_feat, max_seq_len):
        
    output = F.interpolate(v_feat.transpose(0, 1).unsqueeze(0),
                    size=max_seq_len, mode='linear',
                    align_corners=False)
    output = output[0, ...].transpose(0, 1)
    return output

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
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length

