
from random import shuffle
import torch
import numpy as np
from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
from utils.utils import convert_length_to_mask, gene_soft_label, iou_n1, score2d_to_moments_scores
import pandas as pd
from tqdm import tqdm
from models import *
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, configs):
        super(Dataset, self).__init__()
        self.dataset = dataset
        # self.dataset.sort(key=lambda x:x['vid'])

        self.video_features = video_features
        self.max_vlen = configs.model.vlen

        # with open(configs.paths.result_model1_path, mode='rb') as f:
        #     self.model1_result = pickle.load(f)

    def __getitem__(self, index):
        index = index
        record = self.dataset[index]
        vfeat = self.video_features[record['vid']]
        sidx, eidx = int(record['s_ind']), int(record['e_ind'])
        words_id, chars_id = record['w_ids'], record['c_ids']
        bert_id, bert_tmask = record["bert_id"], record["bert_mask"]
        label1d = self.get_dist_idx(sidx, eidx)
        map2d_contrasts = self.get_map2d_contrast(sidx, eidx)
        NER_label = self.get_NER_label(sidx, eidx, vfeat)
        label2d = self.get_label2d(record['s_time'], record['e_time'], record['duration'])

        # label1d_model1 = self.get_label1d_model(index, record['vid'])
        res = {"record": record,
               "max_vlen": self.max_vlen,
               "vfeat": vfeat,
               "words_id": words_id,
               "chars_id": chars_id,
               "bert_id": bert_id,
               "bert_tmask": bert_tmask,
               "label1d": label1d,
               "label2d": label2d,
            #    "label1d_model1": label1d_model1,
               "NER_label": NER_label,
               "map2d_contrast": map2d_contrasts,
               "se_time": [record["s_time"], record["e_time"]],
               "se_frac": [record["s_time"]/record["duration"], record["e_time"]/record["duration"]]
            }
        return res


    def __len__(self):
        return len(self.dataset)

    def get_dist_idx(self, sidx, eidx):
        visual_len = self.max_vlen
        dist_idx = np.zeros((2, visual_len), dtype=np.float32)
        gt_s, gt_e = sidx, eidx
        gt_length = gt_e - gt_s + 1  # make sure length > 0
        dist_idx[0, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
        dist_idx[1, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
        dist_idx[0, dist_idx[0, :] >= 0.8] = 1.
        dist_idx[0, dist_idx[0, :] < 0.1353] = 0.
        dist_idx[1, dist_idx[1, :] >= 0.8] = 1.
        dist_idx[1, dist_idx[1, :] < 0.1353] = 0.
        if (dist_idx[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
            idx = np.argsort(p)
            dist_idx[0, idx[-1]] = 1.
        if (dist_idx[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
            idx = np.argsort(p)
            dist_idx[1, idx[-1]] = 1.
        dist_idx = torch.from_numpy(dist_idx)
        return dist_idx

    def get_map2d_contrast(self, sidx, eidx):
        num_clips = self.max_vlen

        x, y = np.arange(0, sidx + 1., dtype=int), np.arange(eidx - 1, num_clips, dtype=int)
        mask2d_pos = np.zeros((num_clips, num_clips), dtype=bool)
        mask_idx = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        mask2d_pos[mask_idx[:, 0], mask_idx[:, 1]] = 1

        mask2d_neg = np.zeros((num_clips, num_clips), dtype=bool)
        for offset in range(sidx):
            i, j = range(0, sidx - offset), range(offset, sidx)
            mask2d_neg[i, j] = 1
        for offset in range(eidx):
            i, j = range(eidx, num_clips - offset), range(eidx + offset, num_clips)
            mask2d_neg[i, j] = 1
        if np.sum(mask2d_neg) == 0:
            mask2d_neg[0, 0] = 1
            mask2d_neg[num_clips - 1, num_clips - 1] = 1
        return torch.tensor(np.array([mask2d_pos, mask2d_neg]))

    def get_NER_label(self, sidx, eidx, vfeat):
        max_len = self.max_vlen
        cur_max_len = len(vfeat)
        st, et = sidx, eidx
        NER_label = torch.zeros([max_len], dtype=torch.int64) 

        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        NER_label[new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        NER_label[(new_st_r + 1):new_et_l] = 2  # add I-M labels
        NER_label[new_et_l:(new_et_r + 1)] = 3  # add E-M labels

        return NER_label
    
    def get_label2d(self, stime, etime, duration):
        num_clips = self.max_vlen
        moment = torch.as_tensor([stime, etime])
        iou2d = torch.ones(num_clips, num_clips)
        candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
        iou2d = iou_n1(candidates, moment).reshape(num_clips, num_clips)
        return iou2d

        # num_clips = self.max_vlen
        # moment = torch.as_tensor([sidx, eidx])
        # iou2d = torch.ones(num_clips, num_clips)
        # grids = iou2d.nonzero(as_tuple=False)    
        # candidates = grids
        # iou2d = iou_n1(candidates, moment).reshape(num_clips, num_clips)
        # return iou2d

    def get_label1d_model(self, index, vid):
        res = self.model1_result[index]["logit1d"]
        res = torch.from_numpy(np.stack(res))
        # assert self.model1_result[index]["vid"] == vid
        return res



# from utils.dataset_charades import CharadesSTA
from .ActionFormerDataset import ActionFormerDataset

def get_loader(dataset, video_features, configs, loadertype):
    data_set = Dataset(dataset=dataset, video_features=video_features, configs=configs)
    if loadertype == "train":
        shuffle = True
    else:
        shuffle = False
    # collate_fn_VSL
    collate_fn = eval("collate_fn_" + configs.model.name)
    loader = torch.utils.data.DataLoader( 
                                        dataset=data_set, 
                                        batch_size=configs.train.batch_size, 
                                        shuffle=shuffle,
                                        collate_fn=collate_fn
                                        )
    return loader



    # data_set = ActionFormerDataset(dataset=dataset, video_features=video_features, 
    #     is_training = loadertype,
    #     feat_stride = 16, 
    #     num_frames = 16, 
    #     input_dim= 512, 
    #     default_fps = 15, 
    #     downsample_rate = 1, 
    #     max_seq_len = 192,
    #     trunc_thresh =  0.5,
    #     crop_ratio = [0.9, 1.0],
    #     force_upsampling = True)