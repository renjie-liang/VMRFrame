import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock

class OneTeacher(nn.Module):
    def __init__(self, configs, word_vectors):
        super(OneTeacher, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self.configs.model.vlen

        ## ---------- student ------
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors, droprate=droprate)
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim, droprate=droprate)
        self.feat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)
        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        self.predictor = SeqPANPredictor(configs)

        # -------- teacher 0 ----------
        
        self.text_encoder_t0 = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim,  char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors, droprate=droprate)
        self.video_affine_t0 = VisualProjection(visual_dim=configs.model.vdim, dim=dim, droprate=droprate)
        self.feat_encoder_t0 = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.dual_attention_block_1_t0 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2_t0 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)
        self.q2v_attn_t0 = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn_t0 = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat_t0 = CQConcatenate(dim=dim)
        self.match_conv1d_t0 = Conv1D(in_dim=dim, out_dim=4)
        lable_emb_t0 = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb_t0 = torch.nn.init.orthogonal_(lable_emb_t0.data)
        self.label_embs_t0 = nn.Parameter(lable_emb_t0, requires_grad=True)
        self.predictor_t0 = SeqPANPredictor(configs)

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        torch.cuda.synchronize()
        start = time.time()
        B = vmask.shape[0]

        # ----------- teacher ------------
        tfeat_t0 = self.text_encoder_t0(word_ids, char_ids)
        vfeat_t0 = self.video_affine_t0(vfeat_in)

        vfeat_t0 = self.feat_encoder_t0(vfeat_t0)
        tfeat_t0 = self.feat_encoder_t0(tfeat_t0)

        vfeat_ = self.dual_attention_block_1_t0(vfeat_t0, tfeat_t0, vmask, tmask)
        tfeat_ = self.dual_attention_block_1_t0(tfeat_t0, vfeat_t0, tmask, vmask)
        vfeat_t0, tfeat_t0 = vfeat_, tfeat_

        vfeat_ = self.dual_attention_block_2_t0(vfeat_t0, tfeat_t0, vmask, tmask)
        tfeat_ = self.dual_attention_block_2_t0(tfeat_t0, vfeat_t0, tmask, vmask)
        vfeat_t0, tfeat_t0 = vfeat_, tfeat_


        t2v_feat_t0 = self.q2v_attn_t0(vfeat_t0, tfeat_t0, vmask, tmask)
        v2t_feat_t0 = self.v2q_attn_t0(tfeat_t0, vfeat_t0, tmask, vmask)
        fuse_feat_t0 = self.cq_cat_t0(t2v_feat_t0, v2t_feat_t0, tmask)


        match_logits_t0 = self.match_conv1d_t0(fuse_feat_t0)
        match_score_t0 = F.gumbel_softmax(match_logits_t0, tau=0.3)
        soft_label_embs_t0 = torch.matmul(match_score_t0, torch.tile(self.label_embs_t0, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat_t0 = (fuse_feat_t0 + soft_label_embs_t0) * vmask.unsqueeze(2)
        slogits_t0, elogits_t0 = self.predictor_t0(fuse_feat_t0, vmask)


        # #  ----------------------- student ----------------
        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)
        vfeat = self.feat_encoder(vfeat)
        tfeat = self.feat_encoder(tfeat)

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)


        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        slogits, elogits = self.predictor(fuse_feat, vmask)


        torch.cuda.synchronize()
        end = time.time()
        consume_time =  end - start

        return {    "slogits_t0": slogits_t0,
                    "elogits_t0": elogits_t0,
                    "match_score_t0" : match_score_t0,
                    "label_embs_t0" : self.label_embs_t0,

                    "slogits": slogits,
                    "elogits": elogits,
                    "match_score" : match_score,
                    "label_embs" : self.label_embs,

                    "vmask" : vmask,
                    "consume_time": consume_time,
                    }
    

from utils.BaseDataset import BaseDataset, BaseCollate
class OneTeacherDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        return res
    
class OneTeacherCollate(BaseCollate):
    def __call__(self, datas):
        return super().__call__(datas)


import time
from models.loss import lossfun_softloc
def train_engine_OneTeacher(model, data, configs):
    from models.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    label1ds =  data['label1ds']
    vmasks =  data['vmasks']

    slogits_t0 = output["slogits_t0"]
    elogits_t0 = output["elogits_t0"]
    loc_loss_t0 = lossfun_loc(slogits_t0, elogits_t0, label1ds[:, 0, :], label1ds[:, 1, :], vmasks)
    m_loss_t0 = lossfun_match(output["match_score_t0"], output["label_embs_t0"],  data["NER_labels"],  vmasks)
    loss_t0 = loc_loss_t0 + m_loss_t0

    slogits = output["slogits"]
    elogits = output["elogits"]
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], vmasks)
    m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  vmasks)
    loss = loc_loss + m_loss

    # --- KL loss ----
    loss_student_t0 = lossfun_softloc(slogits, elogits, slogits_t0, elogits_t0, vmasks, configs.loss.temperature)

    loss = loss_t0 + loss + loss_student_t0
    return loss, output


def infer_OneTeacher(output, configs):
    from utils.engine import infer_basic
    vmask = output["vmask"]
    res = infer_basic(output["slogits"], output["elogits"], vmask)
    return res