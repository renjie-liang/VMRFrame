
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
        self.dataset.sort(key=lambda x:x['vid'])

        self.video_features = video_features
        self.max_vlen = configs.model.vlen

        with open(configs.paths.result_model1_path, mode='rb') as f:
            self.model1_result = pickle.load(f)

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

        label1d_model1 = self.get_label1d_model(index, record['vid'])
        res = {"record": record,
               "max_vlen": self.max_vlen,
               "vfeat": vfeat,
               "words_id": words_id,
               "chars_id": chars_id,
               "bert_id": bert_id,
               "bert_tmask": bert_tmask,
               "label1d": label1d,
               "label2d": label2d,
               "label1d_model1": label1d_model1,
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
# def collate_fn_VSL(data):
#     records, video_features, word_ids, char_ids, s_inds, e_inds = zip(*data)
#     # process word ids
#     word_ids, _ = pad_seq(word_ids)
#     word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
#     # process char ids
#     char_ids, _ = pad_char_seq(char_ids)
#     char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
#     # process video features
#     vfeats, vfeat_lens = pad_video_seq(video_features)
#     vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
#     vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
#     # process labels
#     max_len = np.max(vfeat_lens)
#     batch_size = vfeat_lens.shape[0]
#     s_labels = np.asarray(s_inds, dtype=np.int64)
#     e_labels = np.asarray(e_inds, dtype=np.int64)
    
#     h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
#     extend = 0.1
#     for idx in range(batch_size):
#         st, et = s_inds[idx], e_inds[idx]
#         cur_max_len = vfeat_lens[idx]
#         extend_len = round(extend * float(et - st + 1))
#         if extend_len > 0:
#             st_ = max(0, st - extend_len)
#             et_ = min(et + extend_len, cur_max_len - 1)
#             h_labels[idx][st_:(et_ + 1)] = 1
#         else:
#             h_labels[idx][st:(et + 1)] = 1

#     # convert to torch tensor
#     vfeats = torch.tensor(vfeats, dtype=torch.float32)
#     word_ids = torch.tensor(word_ids, dtype=torch.int64)
#     char_ids = torch.tensor(char_ids, dtype=torch.int64)
#     s_labels = torch.tensor(s_labels, dtype=torch.int64)
#     e_labels = torch.tensor(e_labels, dtype=torch.int64)
#     h_labels = torch.tensor(h_labels, dtype=torch.int64)

#     tmask = (torch.zeros_like(word_ids) != word_ids).float()
#     vmask = convert_length_to_mask(vfeat_lens)
#     return records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, h_labels






# def collate_fn_CPL(data):
#     records, vfeats_raw, word_ids, char_ids, s_inds, e_inds, max_lens = zip(*data)
#     max_len = max_lens[0]
#     B = len(data)

#     # process word ids
#     word_ids, _ = pad_seq(word_ids)
#     word_ids = np.asarray(word_ids, dtype=np.int32)  # (B, w_seq_len)
#     # process char ids
#     char_ids, _ = pad_char_seq(char_ids)
#     char_ids = np.asarray(char_ids, dtype=np.int32)  # (B, w_seq_len, c_seq_len)
#     # process video features
#     vfeats, vfeat_lens = pad_video_seq(vfeats_raw, max_len)
#     vfeats = np.asarray(vfeats, dtype=np.float32)  # (B, v_seq_len, v_dim)
#     vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (B, )

#     # process labels

#     s_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
#     e_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
#     m_labels = np.zeros(shape=[B, max_len], dtype=np.int32)  # (B, v_seq_len)
#     new_m_labels = np.zeros(shape=[B, max_len, 4], dtype=np.float32)  # (B, v_seq_len)

#     new_s_labels, new_e_labels, new_match_labels = [], [], []

#     for idx in range(B):
#         st, et = s_inds[idx], e_inds[idx]
#         cur_max_len = vfeat_lens[idx]
#         # create classification labels
#         s_labels[idx][0:cur_max_len] = 1e-10
#         e_labels[idx][0:cur_max_len] = 1e-10
#         y = (1 - cur_max_len * 1e-10 - 0.5) / 2
#         s_labels[idx][st] = s_labels[idx][st] + 0.5
#         if st > 0:
#             s_labels[idx][st - 1] = y
#         else:
#             s_labels[idx][st] = s_labels[idx][st] + y
#         if st < cur_max_len - 1:
#             s_labels[idx][st + 1] = y
#         else:
#             s_labels[idx][st] = s_labels[idx][st] + y
#         e_labels[idx][et] = e_labels[idx][et] + 0.5
#         if et > 0:
#             e_labels[idx][et - 1] = y
#         else:
#             e_labels[idx][et] = e_labels[idx][et] + y
#         if et < cur_max_len - 1:
#             e_labels[idx][et + 1] = y
#         else:
#             e_labels[idx][et] = e_labels[idx][et] + y
#         # create matching labels
#         ext_len = 1
#         new_st_l = max(0, st - ext_len)
#         new_st_r = min(st + ext_len, cur_max_len - 1)
#         new_et_l = max(0, et - ext_len)
#         new_et_r = min(et + ext_len, cur_max_len - 1)
#         if new_st_r >= new_et_l:
#             new_st_r = max(st, new_et_l - 1)
#         m_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
#         m_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
#         m_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels


#         new_m_labels[idx, np.where(m_labels[idx]==0), 0] = 1 
#         new_m_labels[idx, np.where(m_labels[idx]==1), 1] = 1 
#         new_m_labels[idx, np.where(m_labels[idx]==2), 2] = 1 
#         new_m_labels[idx, np.where(m_labels[idx]==3), 3] = 1 


#         Ssoft, Esoft, Msoft = gene_soft_label(st, et, cur_max_len, max_len, 0.1)
#         new_s_labels.append(Ssoft)
#         new_e_labels.append(Esoft)
#         new_match_labels.append(Msoft)

#     new_s_labels = np.stack(new_s_labels)
#     new_e_labels = np.stack(new_e_labels)
#     new_match_labels = np.stack(new_match_labels)

#     # convert to torch tensor
#     vfeats = torch.tensor(vfeats, dtype=torch.float32)
#     word_ids = torch.tensor(word_ids, dtype=torch.int64)
#     char_ids = torch.tensor(char_ids, dtype=torch.int64)
#     s_labels = torch.tensor(new_s_labels, dtype=torch.float32)
#     e_labels = torch.tensor(new_e_labels, dtype=torch.float32)
#     m_labels = torch.tensor(new_match_labels, dtype=torch.float32)
    
#     tmask = (torch.zeros_like(word_ids) != word_ids).float()
#     vmask = convert_length_to_mask(vfeat_lens, max_len=max_len)

#     return records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels



    # res = {"record": record,
    #         "max_len": max_len,

    #         "vfeat": vfeat,
    #         "word_ids": word_ids,
    #         "dist_idx": dist_idx,


# from torch.nn.utils.rnn import pad_sequence
# def collate_fn_BAN(batch):
#     visual, text, visual_len, iou2d, duration, timestamp, mask2d_contrast, \
#     start_end_offset, start_end_gt, map_gt, vname, sentence = zip(*batch)
#     text_lens = [len(t) for t in text]
#     text_lens = torch.as_tensor(np.array(text_lens), dtype=torch.int64)
#     video_len = [len(v) for v in visual]
#     video_len = torch.as_tensor(np.array(video_len), dtype=torch.int64)
#     visual = torch.stack(visual)
#     iou2d = torch.stack(iou2d)
#     s_e_distribution = torch.stack(map_gt)
#     mask2d_contrast = torch.stack(mask2d_contrast)

#     start_end_offset = torch.stack(start_end_offset)
#     start_end_gt = torch.stack(start_end_gt)

#     pad_text = pad_sequence(text, batch_first=True, padding_value=0)
#     timestamp = torch.stack(timestamp)
#     duration = torch.as_tensor(np.array(duration), dtype=torch.float)


#     vname = list(vname)
#     sentence = list(sentence)

#     data = {'q_feature': pad_text,
#             'q_len': text_lens,
#             'v_feature': visual,
#             'v_len': video_len,
#             'timestamp': timestamp,
#             'iou2d': iou2d,
#             'start_end_offset': start_end_offset,
#             'start_end_gt': start_end_gt,
#             's_e_distribution': s_e_distribution,
#             'duration': duration,
#             'mask2d_contrast': mask2d_contrast
#             }
#     info = {'vname': vname,
#             'sentence': sentence}
#     return data, info



# from utils.dataset_charades import CharadesSTA

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