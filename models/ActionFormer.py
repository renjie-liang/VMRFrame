import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.ActionFormerlib.meta_archs import PtTransformer

class ActionFormer(nn.Module):
    def __init__(self, configs, word_vec):
        super(ActionFormer, self).__init__()
        actionformer_config = configs.actionformer
        self.net = PtTransformer(**actionformer_config)

    def forward(self, input):
        res = self.net(input)
        return res

from utils.BaseDataset import BaseDataset, BaseCollate
class ActionFormerDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
        self.default_fps = configs.dataprocess.default_fps
        self.force_upsampling = configs.dataprocess.force_upsampling
        self.feat_stride = configs.dataprocess.feat_stride
        self.downsample_rate = configs.dataprocess.downsample_rate
        self.num_frames = configs.dataprocess.num_frames
        self.is_training = loadertype

    def __getitem__(self, index):
        baseres = BaseDataset.__getitem__(self, index)
        feats = baseres["vfeat"]
        record = baseres["record"]

        # feats = self.video_features[record['vid']]
        video_item = {}
        video_item['fps'] = self.default_fps
        video_item['segments'] = np.array(record['se_time'])
        video_item['labels'] = np.array([0])

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float((feats.shape[0] - 1) * self.feat_stride + self.num_frames) / self.max_vlen
            # center the features
            num_frames = feat_stride
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = feats.shape[0]
            assert seq_len <= self.max_vlen
            if self.force_upsampling:
                # reset to max_vlen
                seq_len = self.max_vlen
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = feats.T

        # resize the features if needed
        if (feats.shape[-1] != self.max_vlen) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_vlen,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here

        segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset)
        labels = torch.tensor([[0]])
        # if video_item['segments'] is not None:
        #     segments = torch.from_numpy(
        #         video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
        #     )
        #     labels = torch.from_numpy(video_item['labels'])
        #     # for activity net, we have a few videos with a bunch of missing frames
        #     # here is a quick fix for training
        #     if self.is_training == "train":
        #         vid_len = feats.shape[1] + feat_offset
        #         valid_seg_list, valid_label_list = [], []
        #         for seg, label in zip(segments, labels):
        #             if seg[0] >= vid_len:
        #                 # skip an action outside of the feature map
        #                 continue
        #             # skip an action that is mostly outside of the feature map
        #             ratio = (
        #                 (min(seg[1].item(), vid_len) - seg[0].item())
        #                 / (seg[1].item() - seg[0].item())
        #             )
        #             if ratio >= self.trunc_thresh:
        #                 valid_seg_list.append(seg.clamp(max=vid_len))
        #                 # some weird bug here if not converting to size 1 tensor
        #                 valid_label_list.append(label.view(1))
        #         segments = torch.stack(valid_seg_list, dim=0)
        #         labels = torch.cat(valid_label_list)
        # else:
        #     segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : record['vid'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : record['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames,
                     "record"          : record,
                     "se_time"         : record['se_time']}

        # no truncation is needed
        # truncate the features during training
        # if self.is_training and (segments is not None):
        #     data_dict = truncate_feats(
        #         data_dict, self.max_vlen, self.trunc_thresh, feat_offset, self.crop_ratio
        #     )

        return data_dict


# def collate_fn_ActionFormer(datas):
#     records = {}
#     tmp = []
#     for d in datas:
#         tmp.append(torch.from_numpy(d["se_time"]).squeeze())
#     records["se_fracs"] = tmp
#     return datas, records

class ActionFormerCollate():
    def __call__(self, datas):
        # res =  super().__call__(datas)
        records = {}
        tmp = []
        for d in datas:
            tmp.append(torch.as_tensor(d["se_time"]).squeeze())
        records["se_fracs"] = tmp
        return datas, records

def train_engine_ActionFormer(model, data, configs, runtype):
    for i in data:
        i["feats"] =  i["feats"].to(configs.device)
        i["segments"] =  i["segments"].to(configs.device)
        i["labels"] =  i["labels"].to(configs.device)
    loss, output = model(data)
    loss = loss["reg_loss"]
    return loss, output


def infer_ActionFormer(output, configs):
    res = []
    for i in output:
        res.append(i["segments"][0])
    res = torch.stack(res)
    return res