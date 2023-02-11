import torch
import torch.nn as nn  
import torch.nn.functional as F
import numpy as np
import time
from .BANlib.model import VisualEncoder, QueryEncoder, CQAttention, TemporalDifference, SparseBoundaryCat, SparseMaxPool
from .BANlib.model import DenseMaxPool, Aaptive_Proposal_Sampling, NaivePredictor, PropPositionalEncoding, Adaptive_Prop_Interaction
from .BANlib.model import sequence2mask, temporal_difference_loss, ContrastLoss
from utils.utils import iou_n1


# from transformers import BertModel
# ----------------------------------------
class BAN(nn.Module):
    def __init__(self, cfg, pre_train_emb=None):
        super(BAN, self).__init__()
        self.vlen = cfg.model.vlen
        self.topk = cfg.model.topk
        self.neighbor = cfg.model.neighbor
        self.negative = cfg.model.negative
        self.prop_num = cfg.model.prop_num
        vocab_size = pre_train_emb.shape[0]
        device = cfg.device

        self.visual_encoder = VisualEncoder(cfg.model.vdim, cfg.model.dim, cfg.model.lstm_layer)
        self.query_encoder = QueryEncoder(vocab_size, cfg.model.dim, embed_dim=cfg.model.query_embed_dim,
                                          num_layers=cfg.model.lstm_layer, pre_train_weights=pre_train_emb)
        self.cross_encoder = VisualEncoder(4 * cfg.model.fuse_dim, cfg.model.dim,
                                           cfg.model.lstm_layer)
        self.cqa_att = CQAttention(cfg.model.fuse_dim)
        self.boundary_aware = TemporalDifference(cfg, in_dim=cfg.model.fuse_dim, layer_num=2)
        self.boundary_aggregation = SparseBoundaryCat(cfg.model.pooling_counts, self.vlen, device)
        if cfg.model.sparse_sample:
            self.content_aggregation = SparseMaxPool(cfg.model.pooling_counts, self.vlen, device)
        else:
            self.content_aggregation = DenseMaxPool(self.vlen, device)
        self.map2d_proj = nn.Sequential(
            nn.Linear(3 * cfg.model.fuse_dim, cfg.model.fuse_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False)
        )
        self.prop_sampler = Aaptive_Proposal_Sampling(self.topk, self.neighbor, self.negative, 0.7)
        self.predictor = NaivePredictor(cfg.model.fuse_dim, cfg.model.fuse_dim, intermediate=True)
        self.predictor2 = NaivePredictor(cfg.model.fuse_dim, cfg.model.fuse_dim, intermediate=True)
        self.predictor_offset = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.fuse_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(cfg.model.fuse_dim, 2)
        )
        self.prop_pe = PropPositionalEncoding(cfg.model.fuse_dim, cfg.model.dim)
        self.fc_fuse = nn.Linear(6 * cfg.model.dim, cfg.model.fuse_dim)
        self.contrast_encoder = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.contrast_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model.contrast_dim, cfg.model.contrast_dim)
        )
        self.contrast_encoder_t = nn.Sequential(
            nn.Linear(cfg.model.fuse_dim, cfg.model.contrast_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model.contrast_dim, cfg.model.contrast_dim)
        )
        self.prop_interact = Adaptive_Prop_Interaction(cfg)
        # self.bert = BertModel.from_pretrained('bert-base-cased')

    # def forward(self, data):
    #     data_visual, data_text, video_seq_len, text_seq_len, offset_gt = \
    #         data['v_feature'], data['q_feature'], data['v_len'], data['q_len'],  data['start_end_offset']
    def forward(self, data_visual, data_text, video_seq_len, text_seq_len, offset_gt):
        torch.cuda.synchronize()
        start = time.time()
        

        # feature encoder
        video_feature, clip_feature = self.visual_encoder(data_visual, video_seq_len, self.vlen)
        sentence_feature, word_feature = self.query_encoder(data_text, text_seq_len)
        mask_word = sequence2mask(text_seq_len)
        cat_feature = self.cqa_att(clip_feature, word_feature, mask_word)
        _, fuse_feature = self.cross_encoder(cat_feature, video_seq_len, self.vlen)
        # boundary prediction
        out = self.boundary_aware(fuse_feature)
        hidden_b, hidden_c = out['feature']
        # td = out['td']  # (bs, seq)
        # proposal generation
        map2d_s_e, _ = self.boundary_aggregation(hidden_b.permute(0, 2, 1), hidden_b.permute(0, 2, 1))
        map2d_c, map2d_mask = self.content_aggregation(fuse_feature.permute(0, 2, 1))
        map2d_c = map2d_c.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        map2d_s_e = map2d_s_e.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        map2d_sec = torch.cat([map2d_s_e, map2d_c], dim=-1)
        map2d = self.map2d_proj(map2d_sec)
        # matching prediction
        tmap = self.predictor(map2d)
        # content feature for contrastive learning
        map2d_proj = self.contrast_encoder(map2d_c)
        sen_proj = self.contrast_encoder_t(sentence_feature)
        B, N, D = hidden_c.size()
        score_pred = tmap.sigmoid() * map2d_mask
        score_pred = score_pred.clone().detach()
        prop_feature, pred_s_e, offset_gt, pred_score = \
            self.prop_sampler(score_pred, map2d_mask, map2d, offset_gt, tmap)

        prop_feature = self.prop_pe(prop_feature.view(-1, D), pred_s_e.view(-1, 2))
        prop_num = self.prop_num

        prop_feature = prop_feature.view(B, prop_num, D)
        pred_s_e = pred_s_e.view(B, prop_num, 2)
        offset_gt = offset_gt.view(B, prop_num, 2)
        pred_score = pred_score.view(B, prop_num)
        # proposal interaction and matching score prediction
        prop_feature = self.prop_interact(prop_feature)
        pred = self.predictor2(prop_feature)

        offset = self.predictor_offset(prop_feature)

        torch.cuda.synchronize()
        end = time.time()
        consume_time = end - start

        out = {'tmap': tmap,
               'map2d_mask': map2d_mask,
               'map2d_proj': map2d_proj,
               'sen_proj': sen_proj,
               'coarse_pred': pred_s_e,
               'coarse_pred_round': pred_s_e,
               'final_pred': pred,
               'offset': offset,
               'offset_gt': offset_gt,
               'td':  out['td'],
               'video_seq_len': video_seq_len,
                "consume_time": consume_time,
               }

        # loss = self.loss(out, data, td)
        return out



def collate_fn_BAN(datas):
    from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
    from utils.utils import convert_length_to_mask
    
    records, se_times, se_fracs, vfeats, words_ids = [], [], [], [], []
    dist_idxs, map2d_contrasts, label2ds = [], [], []
    max_vlen = datas[0]["max_vlen"]
    for d in datas:
        records.append(d["record"])
        vfeats.append(d["vfeat"])
        words_ids.append(d["words_id"])
        dist_idxs.append(d["label1d"])
        map2d_contrasts.append(d["map2d_contrast"])
        se_times.append(d["se_time"])
        se_fracs.append(d["se_frac"])
        # label2ds.append(d["label2d"])

    # process word ids
    words_ids, _ = pad_seq(words_ids)
    words_ids = torch.as_tensor(words_ids, dtype=torch.int64)
    tmask = (torch.zeros_like(words_ids) != words_ids).float()
    tlens = torch.sum(tmask, dim=1, keepdim=False, dtype=torch.int64)
    vfeats, vlens = pad_video_seq(vfeats, max_vlen)
    vfeats = torch.stack(vfeats)
    vlens = torch.as_tensor(vlens, dtype=torch.int64)
    dist_idxs = torch.stack(dist_idxs)
    se_times = torch.as_tensor(se_times, dtype=torch.float)
    se_fracs = torch.as_tensor(se_fracs, dtype=torch.float)
    # label2ds = torch.stack(label2ds)

    # process labels
    num_clips = max_vlen
    start_end_offset, iou2ds = [], []
    for recor in records:
        duration = recor["duration"]
        moment = recor["se_time"]
        moment = torch.as_tensor(moment)
        
        iou2d = torch.ones(num_clips, num_clips)
        grids = iou2d.nonzero(as_tuple=False)    
        candidates = grids * duration / num_clips
        iou2d = iou_n1(candidates, moment).reshape(num_clips, num_clips)

        se_offset = torch.ones(num_clips, num_clips, 2)  # not divided by number of clips
        se_offset[:, :, 0] = ((moment[0] - candidates[:, 0]) / duration).reshape(num_clips, num_clips)
        se_offset[:, :, 1] = ((moment[1] - candidates[:, 1]) / duration).reshape(num_clips, num_clips)

        start_end_offset.append(se_offset)
        iou2ds.append(iou2d)

    iou2ds = torch.stack(iou2ds)
    start_end_offset = torch.stack(start_end_offset)
    map2d_contrasts = torch.stack(map2d_contrasts)
    res = {'words_ids': words_ids,
            'tlens': tlens,
            'vfeats': vfeats,
            'vlens': vlens,
            'start_end_offset': start_end_offset,
            'iou2ds': iou2ds,
            # 'label2ds': label2ds,
            'dist_idxs': dist_idxs,
            'map2d_contrasts': map2d_contrasts,
            'se_times': se_times,
            'se_fracs': se_fracs,
            }

    return res, records


def scale(iou, min_iou, max_iou):
    return (iou - min_iou) / (max_iou - min_iou)


def train_engine_BAN(model, data, configs):
    data = {key: value.to(configs.device) for key, value in data.items()}
    out = model(data['vfeats'], data['words_ids'], data['vlens'], data['tlens'], data['start_end_offset'])

    # loss bce
    scores2d, ious2d, mask2d = out['tmap'], data['iou2ds'], out['map2d_mask'],
    ious2d_scaled = scale(ious2d, configs.loss.min_iou, configs.loss.max_iou).clamp(0, 1)
    loss_bce = F.binary_cross_entropy_with_logits(
        scores2d.squeeze().masked_select(mask2d),
        ious2d_scaled.masked_select(mask2d)
    )

    # loss refine
    final_pred = out['final_pred']
    pred_s_e_round = out['coarse_pred_round']
    ious_gt = []
    for i in range(ious2d_scaled.size(0)):
        start = pred_s_e_round[i][:, 0]
        end = pred_s_e_round[i][:, 1] - 1
        final_ious = ious2d_scaled[i][start, end]
        ious_gt.append(final_ious)
    ious_gt = torch.stack(ious_gt)

    loss_refine = F.binary_cross_entropy_with_logits(
        final_pred.squeeze().flatten(),
        ious_gt.flatten()
    )

    # distribute differe

    dist_idxs =  data['dist_idxs']
    td = out['td']
    td_mask = dist_idxs.sum(dim=1)
    loss_td = temporal_difference_loss(td, td_mask)


    # offset loss
    offset_pred, offset_gt = out['offset'], out['offset_gt'] 
    offset_pred = offset_pred.reshape(-1, 2)
    offset_gt = offset_gt.reshape(-1, 2)
    offset_loss_fun = nn.SmoothL1Loss()
    loss_offset = offset_loss_fun(offset_pred[:, 0], offset_gt[:, 0]) + offset_loss_fun(offset_pred[:, 1], offset_gt[:, 1])


    # contrast loss

    map2d_contrasts = data['map2d_contrasts']
    sen_proj, map2d_proj = out['sen_proj'],  out['map2d_proj']
    mask2d_pos = map2d_contrasts[:, 0, :, :]
    mask2d_neg = map2d_contrasts[:, 1, :, :]
    mask2d_pos = torch.logical_and(mask2d, mask2d_pos)
    mask2d_neg = torch.logical_and(mask2d, mask2d_neg)
    loss_contrast = ContrastLoss()(sen_proj, map2d_proj, mask2d_pos, mask2d_neg)


    loss = loss_bce * configs.loss.bce \
         + loss_refine * configs.loss.refine \
         + loss_td * configs.loss.td \
         + loss_offset * configs.loss.offset \
         + loss_contrast * configs.loss.contrast
    return loss, out


def nms(moments, scores, topk=5, thresh=0.5):
    from models.BAN import iou

    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = torch.zeros_like(ranks).bool()
    numel = suppressed.numel()
    count = 0
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    return moments[~suppressed]

# def infer_BAN(output, configs): ## don't consider vmask
#     num_clips = configs.model.vlen
#     nms_thresh=0.7

#     score_pred = output['final_pred'].sigmoid()
#     prop_s_e = output['coarse_pred_round']
#     res = []
#     for idx, score1d in enumerate(score_pred):
#         candidates = prop_s_e[idx] / num_clips
#         moments = nms(candidates, score1d, topk=1, thresh=nms_thresh)
#         res.append(moments[0])
#     res = torch.stack(res)
#     res = res.cpu().numpy()
#     return res
        
def infer_BAN(output, configs):
    vmask = output["video_seq_len"]
 
    outer = torch.triu(output["tmap"], diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask).cpu().numpy()
    efrac = (end_index/vmask).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res