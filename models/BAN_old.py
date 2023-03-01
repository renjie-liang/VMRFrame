import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import numpy as np
import math

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class QueryEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, embed_dim=300, num_layers=1, 
                 bidirection=True, pre_train_weights=None):
        super(QueryEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pre_train_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pre_train_weights))
            self.embedding.weight.requires_grad = False

            self.pad_vec = nn.Parameter(torch.zeros(size=(1, embed_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, embed_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(pre_train_weights, dtype=torch.float32), requires_grad=False)


        self.biLSTM = nn.LSTM(embed_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)

    def forward(self, query_tokens, query_length):

        # query_embedding = self.embedding(query_tokens)
        query_embedding = F.embedding(query_tokens, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                padding_idx=0)


        query_embedding = pack_padded_sequence(query_embedding,
                                               query_length.to('cpu').data.numpy(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # h_0, c_0 is init as zero here
        output, _ = self.biLSTM(query_embedding) # return (out, (h_n,c_n))
        # c_n and h_n: (num_directions, batch, hidden_size)
        # out: (batch, seq_len, num_directions, hidden_size)
        output, query_length_ = pad_packed_sequence(output, batch_first=True)

        # words_feat, _ = self.bert(input_ids= bert_id, attention_mask=bert_mask,return_dict=False)

        q_vector_list = []
        batch_size = query_length_.size(0)
        for i, length in enumerate(query_length_):
            hidden = output[i][0:length]
            q_vector = torch.mean(hidden, dim=0)
            q_vector_list.append(q_vector)
        q_vector = torch.stack(q_vector_list)
        return q_vector, output
        
        
class VisualEncoder(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=512, num_layers=1, bidirection=True):
        super(VisualEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.biLSTM = nn.LSTM(input_dim, self.hidden_dim, num_layers, dropout=0.0,
                              batch_first=True, bidirectional=bidirection)

    def forward(self, visual_data, visual_length, max_seq_len):
        visual_embedding = pack_padded_sequence(visual_data,
                                               visual_length.to('cpu').data.numpy(),
                                               batch_first=True,
                                               enforce_sorted=False)
        # h_0, c_0 is init as zero here
        output, _ = self.biLSTM(visual_embedding) # return (out, (h_n,c_n))
        # c_n and h_n: (num_directions, batch, hidden_size)
        # out: (batch, seq_len, num_directions, hidden_size)
        output, visual_length_ = pad_packed_sequence(output, batch_first=True, total_length=max_seq_len)

        v_vector_list = []
        batch_size = visual_length_.size(0)
        for i, length in enumerate(visual_length_):
            hidden = output[i][0:length]
            v_vector = torch.mean(hidden, dim=0)
            v_vector_list.append(v_vector)
        v_vector = torch.stack(v_vector_list)
        return v_vector, output



def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target + (1 - mask) * (-1e30)


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Qmask):
        # input: batch, seq, hidden
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Qmask = Qmask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(S, dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out  # batch, seq, 4*hidden

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res



def temporaldifference(feature):
    # (bs, seq, hidden)
    feature_rpad = F.pad(feature.permute(0, 2, 1), (0, 1))  # (bs, hidden, seq + 1)
    feature_lpad = F.pad(feature.permute(0, 2, 1), (1, 0))  # (bs, hidden, seq + 1)
    feature_rpad[:, :, -1] = feature.permute(0, 2, 1)[:, :, -1]
    feature_lpad[:, :, 0] = feature.permute(0, 2, 1)[:, :, 0]
    td_1 = feature_rpad[:, :, 1:] - feature.permute(0, 2, 1)  # (bs, hidden, seq)
    td_2 = feature_lpad[:, :, :-1] - feature.permute(0, 2, 1)  # (bs, hidden, seq)
    td = td_1.square() + td_2.square()
    td = td.permute(0, 2, 1)  # (bs, seq, hidden)
    return td


class TemporalDifference(nn.Module):
    def __init__(self, config, in_dim=None, model_type='lstm', layer_num=1):
        super().__init__()
        self.split_dim = config.model.fuse_dim
        if in_dim == None:
            in_dim = self.split_dim
        self.model_type = model_type
        if model_type == 'lstm':
            self.feature_transform_b = nn.LSTM(in_dim, self.split_dim, layer_num,
                                               batch_first=True, bidirectional=True)
            self.feature_transform_c = nn.LSTM(in_dim, self.split_dim, layer_num,
                                               batch_first=True, bidirectional=True)
            self.feature_proj_b = nn.Sequential(
                nn.Linear(2 * self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.droprate, inplace=False)
            )
            self.feature_proj_c = nn.Sequential(
                nn.Linear(2 * self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.droprate, inplace=False)
            )
        elif model_type == 'cnn':
            self.feature_transform_b = torch.nn.Sequential(
                nn.Conv1d(self.split_dim, self.split_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.split_dim),
                nn.ReLU()
            )
            self.feature_transform_c = torch.nn.Sequential(
                nn.Conv1d(self.split_dim, self.split_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.split_dim),
                nn.ReLU()
            )
            self.feature_proj_b = nn.Sequential(
                nn.Linear(self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.droprate, inplace=False)
            )
            self.feature_proj_c = nn.Sequential(
                nn.Linear(self.split_dim, self.split_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.model.droprate, inplace=False)
            )
        else:
            raise NotImplementedError("sequence model in TD not implemented!")

    def forward(self, visual_input):
        # (B, T, D)
        if self.model_type == 'lstm':
            hidden_b, _ = self.feature_transform_b(visual_input)
            hidden_c, _ = self.feature_transform_c(visual_input)
        elif self.model_type == 'cnn':
            hidden_b = self.feature_transform_b(visual_input.permute(0, 2, 1)).permute(0, 2, 1)
            hidden_c = self.feature_transform_c(visual_input.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_b = self.feature_proj_b(hidden_b)
        hidden_c = self.feature_proj_c(hidden_c)
        td = temporaldifference(hidden_b)  # (bs, seq, hidden)
        td = td.sum(dim=-1)
        return {'feature': [hidden_b, hidden_c],
                'td': td}


import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import numpy as np


class DenseMaxPool(nn.Module):
    def __init__(self, N, device='cpu'):
        super().__init__()
        self.identity = nn.Identity()
        self.pool = nn.MaxPool1d(2, stride=1)
        self.seq_len = N
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1
        maskij = []
        for idx in range(N):
            start_idxs = [s_idx for s_idx in range(0, N - idx, 1)]
            end_idxs = [s_idx + idx for s_idx in start_idxs]
            mask2d[start_idxs, end_idxs] = 1
            # mask2d[:, :, start_idxs, end_idxs] = 1
            maskij.append((start_idxs, end_idxs))
        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        for idx in range(self.seq_len):
            if idx == 0:
                x = self.identity(x)
            else:
                x = self.pool(x)
            start_idxs, end_idxs = self.maskij[idx]
            map2d[:, :, start_idxs, end_idxs] = x
        return map2d, self.mask2d


class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N, device='cpu'):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset), range(offset, N)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for i in range(1, len(pooling_counts)):
            poolers.extend(
                [nn.MaxPool1d(2 * i + 1, 1) for _ in range(pooling_counts[i])]
            )
        self.mask2d = mask2d
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d, self.mask2d


class SparseBoundaryCat(nn.Module):
    def __init__(self, pooling_counts, N, device='cpu'):
        super(SparseBoundaryCat, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset), range(offset, N)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for i in range(1, len(pooling_counts)):
            poolers.extend(
                [nn.MaxPool1d(2 * i + 1, 1) for _ in range(pooling_counts[i])]
            )
        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, start, end):
        B, D, N = start.shape
        map2d = start.new_zeros(B, 2 * D, N, N)
        map2d[:, :, range(N), range(N)] = torch.cat([start, end], dim=1)
        for (i, j) in self.maskij:
            tmp = torch.cat((start[:, :, i], end[:, :, j]), dim=1)
            map2d[:, :, i, j] = tmp
        return map2d, self.mask2d


class Aggregation_center(nn.Module):
    def __init__(self, config, device):
        super(Aggregation_center, self).__init__()
        hidden = config.vilt.fuse_dim
        max_video_seq_len = config.model.video_seq_len
        self.content_aggregation = SparseMaxPool(config.model.pooling_counts, max_video_seq_len, device)
        # self.content_aggregation = DenseMaxPool(max_video_seq_len, device)

    def forward(self, hidden_c):
        map2d_c, map2d_mask = self.content_aggregation(hidden_c.permute(0, 2, 1))
        map2d_c = map2d_c.permute(0, 2, 3, 1)  # (batch, seq, seq, hidden)
        return map2d_c, map2d_mask








import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import math



def iou(candidates, gt):
    '''
    candidates: (prop_num, 2)
    gt: (2, )
    '''
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union



def proposal_selection_with_negative(moments, scores, thresh=0.5, topk=5, neighbor=16, negative=16):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = torch.zeros_like(ranks).bool()
    select = torch.zeros_like(ranks).bool()
    numel = suppressed.numel()
    count = 0
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i] = True
        select[i] = True
        ind_sel = mask.nonzero(as_tuple=False).squeeze(-1)
        if ind_sel.numel() != 0:
            #             suppressed[i + 1:][ind_sel] = True
            ind_sel = ind_sel[:neighbor]
            select[i + 1:][ind_sel] = True
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    total_num = topk * (neighbor + 1)
    if select.sum() < total_num:
        moments_sel_pos = moments[~suppressed][:int(total_num - select.sum())]
        moments_sel_neg = torch.flip(moments[~suppressed], dims=(0,))[:negative]
        moments_sel = torch.cat([moments_sel_neg, moments_sel_pos, moments[select]], dim=0)
    else:
        moments_sel_neg = torch.flip(moments[~suppressed], dims=(0,))[:negative]
        moments_sel = torch.cat([moments_sel_neg, moments[select]], dim=0)
    return moments_sel


class Aaptive_Proposal_Sampling(nn.Module):
    def __init__(self, topk=5, neighbor=16, negative=16, thresh=0.5):
        super().__init__()
        self.topk = topk
        self.neighbor = neighbor
        self.thresh = thresh
        self.negative = negative

    def forward(self, score_pred, map2d_mask, map2d, offset_gt, tmap):
        pred_s_e = []
        prop_lists = []
        offset_gt_list = []
        pred_score = []
        for b in range(score_pred.size(0)):
            grids = map2d_mask.nonzero(as_tuple=False)
            scores = score_pred[b][grids[:, 0], grids[:, 1]]
            grids[:, 1] += 1
            prop_s_e_topk = proposal_selection_with_negative(grids, scores,
                                                             thresh=self.thresh,
                                                             topk=self.topk,
                                                             neighbor=self.neighbor,
                                                             negative=self.negative)
            segs = map2d[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1]
            prop_lists.append((segs))
            offset_gt_list.append(offset_gt[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1, :])
            pred_s_e.append(prop_s_e_topk)
            pred_score.append(tmap[b][prop_s_e_topk[:, 0], prop_s_e_topk[:, 1] - 1])
        prop_feature = torch.cat(prop_lists, dim=0)  # (bs x prop_num, dim)
        pred_s_e = torch.cat(pred_s_e, dim=0)  # (bs x prop_num, 2)
        offset_gt = torch.cat(offset_gt_list, dim=0)  # (bs x prop_num, 2)
        pred_score = torch.cat(pred_score, dim=0)  # (bs x prop_num, 2)
        return prop_feature, pred_s_e, offset_gt, pred_score

import torch
from torch import nn


class NaivePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, intermediate=True, drop_rate=0.1):
        super().__init__()
        if intermediate:
            self.pred = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(drop_rate, inplace=False),
                    nn.Linear(hidden_size, 1)
                )
        else:
            self.pred = nn.Linear(input_size, 1)

    def forward(self, x):
        tmap_logit = self.pred(x)
        return tmap_logit.squeeze(-1)

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.nn.utils.rnn import pad_sequence  # used in pad_collate
import numpy as np
import logging
import time
import math


class PropPositionalEncoding(nn.Module):
    def __init__(self, dim_in=512, dim_emb=256, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, dim_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_emb, 2).float() * (-math.log(10000.0) / dim_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(dim_in + 2*dim_emb, dim_in)

    def forward(self, x, prop_s_e):
        # x: (B, N, hidden) or (B*N, hidden)
        if x.dim() == 3:
            B, N, D = x.size()
            all_num = B * N
            x = x.view(-1, D)  # (prop_num, D)
        else:
            all_num, D = x.size()
        s, e = prop_s_e[:, 0], prop_s_e[:, 1]
        pe_table = self.pe.repeat(all_num, 1, 1)
        pos_s = pe_table[torch.arange(all_num), s, :]
        pos_e = pe_table[torch.arange(all_num), e-1, :]
        # pos_s = pos_s.view(N, -1)
        # pos_e = pos_e.view(N, -1)
        x = torch.cat([x, pos_s, pos_e], dim=-1)
        x = self.fc(x)
        if x.dim() == 3:
            x = x.view(B, N, D)
        return x


def sequence2mask(seq_len, maxlen=None):
    # seq_len: (batch, 1) or (batch, )
    seq_len = seq_len.squeeze()
    seq_len = seq_len.unsqueeze(-1)
    batch_size = len(seq_len)
    if maxlen is None:
        maxlen = seq_len.max()
    tmp1 = torch.arange(0, maxlen, device=seq_len.device).unsqueeze(0)
    tmp2 = seq_len.type(tmp1.type())
    tmp2 = tmp2.expand(batch_size, maxlen)
    mask = torch.ge(tmp1, tmp2)
    return ~mask


def Prepare_logger(log_name, print_console=True):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if print_console:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = log_name + date + '.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

import torch
from torch import nn
import numpy as np
import math
from copy import deepcopy
import torch.nn.functional as F


class LearnPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(LearnPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.pos_embed.weight)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q):
        bsz_q, d_model, q_frm = q.shape
        assert q_frm == self.pos_embed.weight.shape[0], (q_frm, self.pos_embed.weight.shape)
        q_pos = self.pos_embed.weight.clone()
        q_pos = q_pos.unsqueeze(0)
        q_pos = q_pos.expand(bsz_q, q_frm, d_model).transpose(1, 2)
        # q_pos = q_pos.contiguous().view(bsz_q, q_frm, n_head, d_k)
        q = q + q_pos
        return self.dropout(q)


def adaptive_graph_feature(x):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    x = x.transpose(2, 1).contiguous()
    tmp1 = x.unsqueeze(1).repeat(1, num_points, 1, 1)
    tmp2 = x.unsqueeze(2).repeat(1, 1, num_points, 1)
    tmp3 = tmp1 - tmp2
    feature = torch.cat((tmp3, tmp2), dim=3).permute(0, 3, 1, 2)
    return feature


class AdaptiveGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(2 * config.gcn.hidden_size, config.gcn.hidden_size, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        B = x.size(0)
        x_f = adaptive_graph_feature(x)  # (B, D, N, N)
        out = self.fc(x_f)  # edge convolution on semantic graph
        out = out.max(dim=-1, keepdim=False)[0]
        return out


class Adaptive_Prop_Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gcn_layer = nn.ModuleList([deepcopy(AdaptiveGCN(config))
                                        for _ in range(config.gcn.num_blocks)])

    def forward(self, prop_feature):
        # (B, N, D) -> (B, D, N)
        prop_feature = prop_feature.transpose(1, 2)
        # encode
        for layer in self.gcn_layer:
            prop_feature = layer(prop_feature)
        return prop_feature.transpose(1, 2)  # (B, D, N) -> (B, N, D)

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
# from utils import contrast_selection

def batch_iou(candidates, gt):
    '''
    candidates: (batch, prop_num, 2)
    gt: (batch, 2)
    '''
    bs, prop_num, _ = candidates.size()
    gt = gt.unsqueeze(1).repeat(1, prop_num, 1)

    start, end = candidates[:, :, 0], candidates[:, :, 1]  # batch, prop_num
    s, e = gt[:, :, 0].float(), gt[:, :, 1].float()  # batch, prop_num
    inter = end.min(e) - start.max(s)  # batch, prop_num
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def sim(x, y):
    '''
    compute dot product similarity
    :param x: (1,  hidden)
    :param y: (batch,  hidden)
    '''
    normx = torch.linalg.norm(x, dim=-1)
    normy = torch.linalg.norm(y, dim=-1)
    x_norm = x / (normx.unsqueeze(-1) + 1e-8)
    y_norm = y / (normy.unsqueeze(-1) + 1e-8)
    return torch.matmul(x_norm, y_norm.T)


class ContrastLoss(nn.Module):
    def __init__(self, margin=1, tao=1., neg_ratio=20.):
        super().__init__()
        self.margin = margin
        self.tao = tao
        self.neg_ratio = neg_ratio

    def forward(self, pos_query, tmap, mask2d_pos, mask2d_neg):
        '''
        :param tmap: (batch, seq, seq, hidden)
        :param mask2d_pos: (batch, seq, seq)
        :param mask2d_neg: (batch, seq, seq)
        :param ious: (batch, seq, seq)
        :return:
        '''
        loss = []
        for i in range(tmap.size(0)):
            # masked selection
            tmp1 = tmap[i][mask2d_pos[i], :]
            tmp2 = tmap[i][mask2d_neg[i], :]
            pos = tmp1 / torch.linalg.norm(tmp1, dim=-1).unsqueeze(-1)
            neg = tmp2 / torch.linalg.norm(tmp2, dim=-1).unsqueeze(-1)
            if pos.size(0) == 0 or neg.size(0) == 0:
                continue
            positive_sim = sim(pos_query[i].unsqueeze(0), pos)  # (1, pos_num)
            negative_sim = sim(pos_query[i].unsqueeze(0), neg)  # (1, neg_num)
            all_sim = torch.cat([positive_sim, negative_sim], dim=-1)  # (1, neg_num+pos_num)
            numerator = torch.exp(positive_sim / self.tao)  # (1, pos_num)
            numerator = numerator.sum(dim=-1)
            denominator = torch.exp(all_sim / self.tao).sum(dim=-1)  # (1, 1)
            tmp = -torch.log(numerator / (denominator + 1e-8))
            loss.append(tmp)
        return sum(loss) / len(loss)


def temporal_difference_loss(td, position_mask):
    '''
    td: (bs, seq)
    position_mask: (bs, seq), smoothed scores for start/end confidence
    '''
    td = td.softmax(dim=-1)
    numerator = position_mask * torch.log(td)
    numerator = numerator.sum(dim=-1)
    denominator = position_mask.sum(dim=-1)
    loss = -numerator / (denominator + 1e-8)
    return loss.mean()
    

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
        self.bert = BertModel.from_pretrained('bert-base-cased')

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
    dist_idxs, map2d_contrasts = [], []
    max_vlen = datas[0]["max_vlen"]
    for d in datas:
        records.append(d["record"])
        vfeats.append(d["vfeat"])
        words_ids.append(d["words_id"])
        dist_idxs.append(d["label1d"])
        map2d_contrasts.append(d["map2d_contrast"])
        se_times.append(d["se_time"])
        se_fracs.append(d["se_frac"])
        
    # word_ids = data["word_ids"]
    # bert_id = torch.vstack(data["bert_id"])
    # bert_mask = torch.vstack(data["bert_mask"])

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
        iou2d = iou(candidates, moment).reshape(num_clips, num_clips)

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
            # 'bert_id': bert_id,
            # 'bert_mask': bert_mask,

            # 'timestamp': timestamp,
            'iou2ds': iou2ds,
            # 'start_end_gt': start_end_gt,
            'dist_idxs': dist_idxs,
            # 'duration': duration,
            'map2d_contrasts': map2d_contrasts,
            'se_times': se_times,
            'se_fracs': se_fracs,
            # 'vname': vname,
            # 'sentence': sentence
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
    from models.BAN import temporal_difference_loss

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
    from models.BAN import ContrastLoss

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