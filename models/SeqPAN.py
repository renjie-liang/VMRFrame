import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock

class SeqPAN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(SeqPAN, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self.configs.model.vlen

        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)
        self.tfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
                                       
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,
                                             droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,
                                              max_pos_len=max_pos_len, droprate=droprate)


        self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)



        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)

        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        
        self.predictor = SeqPANPredictor(configs)



    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        B = vmask.shape[0]

        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)

        vfeat = self.vfeat_encoder(vfeat)
        tfeat = self.tfeat_encoder(tfeat)


        vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_


        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)

        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        slogits, elogits = self.predictor(fuse_feat, vmask)

        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : vmask,
                    "match_score" : match_score,
                    "label_embs" : self.label_embs,
                    }


def collate_fn_SeqPAN(datas):
    from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
    from utils.utils import convert_length_to_mask


    records, se_times, se_fracs = [], [], []
    vfeats, words_ids, chars_ids = [], [], []
    dist_idxs, NER_labels = [], []
    max_vlen = datas[0]["max_vlen"]
    for d in datas:
        records.append(d["record"])
        vfeats.append(d["vfeat"])
        words_ids.append(d["words_id"])
        dist_idxs.append(d["dist_idx"])
        se_times.append(d["se_time"])
        se_fracs.append(d["se_frac"])
        chars_ids.append(d["chars_id"])
        NER_labels.append(d['NER_label'])
    # process text
    words_ids, _ = pad_seq(words_ids)
    words_ids = torch.as_tensor(words_ids, dtype=torch.int64)
    tmasks = (torch.zeros_like(words_ids) != words_ids).float()
    
    chars_ids, _ = pad_char_seq(chars_ids)
    chars_ids = torch.as_tensor(chars_ids, dtype=torch.int64)

    # process video 
    vfeats, vlens = pad_video_seq(vfeats, max_vlen)
    vfeats = torch.stack(vfeats)
    vlens = torch.as_tensor(vlens, dtype=torch.int64)
    vmasks = convert_length_to_mask(vlens, max_len=max_vlen)
    
    # process label
    dist_idxs = torch.stack(dist_idxs)
    NER_labels = torch.stack(NER_labels)
    
    se_times = torch.as_tensor(se_times, dtype=torch.float)
    se_fracs = torch.as_tensor(se_fracs, dtype=torch.float)

    res = {'words_ids': words_ids,
            'char_ids': chars_ids,
            'tmasks': tmasks,

            'vfeats': vfeats,
            'vmasks': vmasks,

            # labels
            'dist_idxs': dist_idxs,
            'NER_labels': NER_labels,

            # evaluate
            'se_times': se_times,
            'se_fracs': se_fracs,
            # 'vname': vname,
            # 'sentence': sentence
            }

    return res, records



def train_engine_SeqPAN(model, data, configs):
    from models.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    slogits = output["slogits"]
    elogits = output["elogits"]

    dist_idxs =  data['dist_idxs']
    loc_loss = lossfun_loc(slogits, elogits, dist_idxs[:, 0, :], dist_idxs[:, 1, :], data['vmasks'])
    m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  data['vmasks'])

    loss =loc_loss #+ m_loss
    return loss, output


def infer_SeqPAN(output, configs):
    from utils.engine import infer_basic

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    sfrac, efrac = infer_basic(start_logits, end_logits, vmask)

    res = np.stack([sfrac, efrac]).T
    return res