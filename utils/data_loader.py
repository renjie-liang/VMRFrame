
from random import shuffle
import torch
import numpy as np
from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
from utils.utils import convert_length_to_mask, gene_soft_label

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, max_len):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.dataset.sort(key=lambda x:x['vid'])

        self.video_features = video_features
        self.max_len = max_len
    def __getitem__(self, index):
        index = index + 1679
        record = self.dataset[index]
        video_feature = self.video_features[record['vid']]
        s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
        word_ids, char_ids = record['w_ids'], record['c_ids']
        max_len = self.max_len
        return record, video_feature, word_ids, char_ids, s_ind, e_ind, max_len

    def __len__(self):
        return len(self.dataset)


def collate_fn_VSL(data):
    records, video_features, word_ids, char_ids, s_inds, e_inds = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_:(et_ + 1)] = 1
        else:
            h_labels[idx][st:(et + 1)] = 1

    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)

    tmask = (torch.zeros_like(word_ids) != word_ids).float()
    vmask = convert_length_to_mask(vfeat_lens)
    return records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, h_labels



def collate_fn_SeqPAN(data):

    records, vfeats_raw, word_ids, char_ids, s_inds, e_inds, max_lens = zip(*data)
    max_len = max_lens[0]
    B = len(data)

    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (B, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (B, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(vfeats_raw, max_len)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (B, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (B, )

    # process labels

    s_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
    e_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
    m_labels = np.zeros(shape=[B, max_len], dtype=np.int32)  # (B, v_seq_len)
    new_m_labels = np.zeros(shape=[B, max_len, 4], dtype=np.float32)  # (B, v_seq_len)

    new_s_labels, new_e_labels, new_match_labels = [], [], []

    for idx in range(B):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        # create classification labels
        s_labels[idx][0:cur_max_len] = 1e-10
        e_labels[idx][0:cur_max_len] = 1e-10
        y = (1 - cur_max_len * 1e-10 - 0.5) / 2
        s_labels[idx][st] = s_labels[idx][st] + 0.5
        if st > 0:
            s_labels[idx][st - 1] = y
        else:
            s_labels[idx][st] = s_labels[idx][st] + y
        if st < cur_max_len - 1:
            s_labels[idx][st + 1] = y
        else:
            s_labels[idx][st] = s_labels[idx][st] + y
        e_labels[idx][et] = e_labels[idx][et] + 0.5
        if et > 0:
            e_labels[idx][et - 1] = y
        else:
            e_labels[idx][et] = e_labels[idx][et] + y
        if et < cur_max_len - 1:
            e_labels[idx][et + 1] = y
        else:
            e_labels[idx][et] = e_labels[idx][et] + y
        # create matching labels
        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        m_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        m_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
        m_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels


        new_m_labels[idx, np.where(m_labels[idx]==0), 0] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==1), 1] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==2), 2] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==3), 3] = 1 


        Ssoft, Esoft, Msoft = gene_soft_label(st, et, cur_max_len, max_len, 0.1)
        new_s_labels.append(Ssoft)
        new_e_labels.append(Esoft)
        new_match_labels.append(Msoft)

    new_s_labels = np.stack(new_s_labels)
    new_e_labels = np.stack(new_e_labels)
    new_match_labels = np.stack(new_match_labels)

    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(new_s_labels, dtype=torch.float32)
    e_labels = torch.tensor(new_e_labels, dtype=torch.float32)
    m_labels = torch.tensor(new_match_labels, dtype=torch.float32)
    
    tmask = (torch.zeros_like(word_ids) != word_ids).float()
    vmask = convert_length_to_mask(vfeat_lens, max_len=max_len)

    return records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels




def collate_fn_CPL(data):

    records, vfeats_raw, word_ids, char_ids, s_inds, e_inds, max_lens = zip(*data)
    max_len = max_lens[0]
    B = len(data)

    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (B, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (B, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(vfeats_raw, max_len)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (B, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (B, )

    # process labels

    s_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
    e_labels = np.zeros(shape=[B, max_len], dtype=np.float32)
    m_labels = np.zeros(shape=[B, max_len], dtype=np.int32)  # (B, v_seq_len)
    new_m_labels = np.zeros(shape=[B, max_len, 4], dtype=np.float32)  # (B, v_seq_len)

    new_s_labels, new_e_labels, new_match_labels = [], [], []

    for idx in range(B):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        # create classification labels
        s_labels[idx][0:cur_max_len] = 1e-10
        e_labels[idx][0:cur_max_len] = 1e-10
        y = (1 - cur_max_len * 1e-10 - 0.5) / 2
        s_labels[idx][st] = s_labels[idx][st] + 0.5
        if st > 0:
            s_labels[idx][st - 1] = y
        else:
            s_labels[idx][st] = s_labels[idx][st] + y
        if st < cur_max_len - 1:
            s_labels[idx][st + 1] = y
        else:
            s_labels[idx][st] = s_labels[idx][st] + y
        e_labels[idx][et] = e_labels[idx][et] + 0.5
        if et > 0:
            e_labels[idx][et - 1] = y
        else:
            e_labels[idx][et] = e_labels[idx][et] + y
        if et < cur_max_len - 1:
            e_labels[idx][et + 1] = y
        else:
            e_labels[idx][et] = e_labels[idx][et] + y
        # create matching labels
        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        m_labels[idx][new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        m_labels[idx][(new_st_r + 1):new_et_l] = 2  # add I-M labels
        m_labels[idx][new_et_l:(new_et_r + 1)] = 3  # add E-M labels


        new_m_labels[idx, np.where(m_labels[idx]==0), 0] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==1), 1] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==2), 2] = 1 
        new_m_labels[idx, np.where(m_labels[idx]==3), 3] = 1 


        Ssoft, Esoft, Msoft = gene_soft_label(st, et, cur_max_len, max_len, 0.1)
        new_s_labels.append(Ssoft)
        new_e_labels.append(Esoft)
        new_match_labels.append(Msoft)

    new_s_labels = np.stack(new_s_labels)
    new_e_labels = np.stack(new_e_labels)
    new_match_labels = np.stack(new_match_labels)

    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(new_s_labels, dtype=torch.float32)
    e_labels = torch.tensor(new_e_labels, dtype=torch.float32)
    m_labels = torch.tensor(new_match_labels, dtype=torch.float32)
    
    tmask = (torch.zeros_like(word_ids) != word_ids).float()
    vmask = convert_length_to_mask(vfeat_lens, max_len=max_len)

    return records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels



def random_interval(se, proportion):
    import random
    s, e = se
    duration_ = proportion*(e - s)
    e_random = e - duration_
    s_ = s + (e_random-s) * random.random()
    e_ = s_ + duration_
    return [s_, e_]

def collate_fn_BAN(data):
    from models.BAN import iou
    records, vfeats_raw, word_ids, char_ids, s_inds, e_inds, max_lens = zip(*data)
    max_vlen = max_lens[0]

    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = torch.as_tensor(word_ids, dtype=torch.int64)
    tmask = (torch.zeros_like(word_ids) != word_ids).float()
    tlens = torch.sum(tmask, dim=1, keepdim=False, dtype=torch.int64)
    vfeats, vlens = pad_video_seq(vfeats_raw, max_vlen)
    vfeats = torch.stack(vfeats)
    # vfeats = torch.as_tensor(vfeats, dtype=torch.float32)
    vlens = torch.as_tensor(vlens, dtype=torch.int64)

    # process labels
    num_clips = max_vlen
    start_end_offset, iou2ds = [], []
    for recor in records:
        duration = recor["duration"]
        moment = recor["s_time"], recor["e_time"]
        moment = random_interval(moment, 0.7)
        moment = torch.as_tensor(moment)
        
        iou2d = torch.ones(num_clips, num_clips)
        grids = iou2d.nonzero(as_tuple=False)    
        candidates = grids * duration / num_clips
        iou2d = iou(candidates, moment).reshape(num_clips, num_clips)


        se_offset = torch.ones(num_clips, num_clips, 2)  # not divided by number of clips
        se_offset[:, :, 0] = ((moment[0] - candidates[:, 0]) / duration).reshape(num_clips, num_clips)
        se_offset[:, :, 1] = ((moment[1] - candidates[:, 1]) / duration).reshape(num_clips, num_clips)

        start_end_offset.append(se_offset)
        iou2ds.append(iou2d)

    iou2ds = torch.stack(iou2ds)
    start_end_offset = torch.stack(start_end_offset)
    data = {'word_ids': word_ids,
            'tlens': tlens,
            'vfeats': vfeats,
            'vlens': vlens,
            'start_end_offset': start_end_offset,

            # 'timestamp': timestamp,
            'iou2ds': iou2ds,
            # 'start_end_gt': start_end_gt,
            # 's_e_distribution': s_e_distribution,
            # 'duration': duration,
            # 'mask2d_contrast': mask2d_contrast,
            # 'vname': vname,
            # 'sentence': sentence
            }

    return records, data

def get_loader(dataset, video_features, configs, loadertype):
    data_set = Dataset(dataset=dataset, video_features=video_features, max_len=configs.model.vlen)
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