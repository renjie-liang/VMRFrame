import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock
from utils.utils import load_pickle, iou_batch

class MultiTeacher(nn.Module):
    def __init__(self, configs, word_vectors):
        super(MultiTeacher, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self.configs.model.vlen
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)

                                       
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=2, max_pos_len=max_pos_len, droprate=droprate)


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
        torch.cuda.synchronize()
        start = time.time()
        
        B = vmask.shape[0]
        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)

        vfeat = self.vfeat_encoder(vfeat)
        tfeat = self.vfeat_encoder(tfeat)


        vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        vfeat, tfeat = vfeat_, tfeat_


        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)

        # f_tfeat = torch.max(tfeat * tmask[1, 1, None], dim=1)[0]
        # fuse_feat = vfeat * f_tfeat.unsqueeze(1)

        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        slogits, elogits = self.predictor(fuse_feat, vmask)

        torch.cuda.synchronize()
        end = time.time()
        consume_time = end - start

        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : vmask,
                    "match_score" : match_score,
                    "label_embs" : self.label_embs,
                    "consume_time": consume_time,
                    }


from utils.BaseDataset import BaseDataset, BaseCollate
class MultiTeacherDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
        self.t0_result = ""
        self.loadertype = loadertype
        self.logits_t = []
        # if loadertype == "train":
        #     for i in range(len(configs.loss.multiteacher)):
        #         self.logits_t.append(load_pickle(configs.loss.multiteacher[i].t_path))
        self.logits_t0 = load_pickle(configs.loss.t0_path)
        self.logits_t1 = load_pickle(configs.loss.t1_path)
        self.logits_t2 = load_pickle(configs.loss.t2_path)

        
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        label1d_t = []
        if self.loadertype == "train":
            # for i in range(len(self.logits_t)):
            #     label1d_t.append(self.load_label1d_teach(self.logits_t[i], index, res['vid'], res['vfeat'].shape[0]))
            # res["label1d_t"] = label1d_t

            label1d_t0 = self.load_label1d_teach(self.logits_t0, index, res['vid'], res['vfeat'].shape[0])
            res["label1d_t0"] = label1d_t0
            label1d_t1 = self.load_label1d_teach(self.logits_t1, index, res['vid'], res['vfeat'].shape[0])
            res["label1d_t1"] = label1d_t1
            label1d_t2 = self.load_label1d_teach(self.logits_t2, index, res['vid'], res['vfeat'].shape[0])
            res["label1d_t2"] = label1d_t2
        return res

class MultiTeacherCollate(BaseCollate):
    def __call__(self, datas):
        res, records = super().__call__(datas)

        if "label1d_t0" in datas[0].keys():
            label1d_t0s = []
            for d in datas:
                label1d_t0s.append(d["label1d_t0"])
            res["label1d_t0s"] = torch.stack(label1d_t0s)

        if "label1d_t1" in datas[0].keys():
            label1d_t1s = []
            for d in datas:
                label1d_t1s.append(d["label1d_t1"])
            res["label1d_t1s"] = torch.stack(label1d_t1s)

        if "label1d_t2" in datas[0].keys():
            label1d_t2s = []
            for d in datas:
                label1d_t2s.append(d["label1d_t2"])
            res["label1d_t2s"] = torch.stack(label1d_t2s)
        return res, records



def calculate_adapt_cof(T_label, GT_label):
    ts = torch.argmax(T_label[:,0,:], dim=1)
    te = torch.argmax(T_label[:,1,:], dim=1)
    tse = torch.stack([ts, te])
    gts = torch.argmax(GT_label[:,0,:], dim=1)
    gte = torch.argmax(GT_label[:,1,:], dim=1)
    gtse = torch.stack([gts, gte])
    res = iou_batch(tse, gtse)
    return res


import time
from models.loss import lossfun_softloc

def train_engine_MultiTeacher(model, data, configs, runtype):
    from models.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    label1ds =  data['label1ds']
    vmasks =  data['vmasks']
    slogits = torch.sigmoid(output["slogits"])
    elogits = torch.sigmoid(output["elogits"])
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], vmasks)
    # m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  data['vmasks'])
    loss = loc_loss

    # t0
    if runtype == "train":
        label1d_t0s = data['label1d_t0s']
        loss_student_t0 = lossfun_softloc(slogits, elogits, label1d_t0s[:, 0, :], label1d_t0s[:, 1, :], vmasks, configs.loss.t0_temperature)
        loss_student_t0 = torch.mean(calculate_adapt_cof(label1d_t0s, label1ds) * loss_student_t0)
        loss += loss_student_t0 * configs.loss.t0_cof

        label1d_t1s = data['label1d_t1s']
        loss_student_t1 = lossfun_softloc(slogits, elogits, label1d_t1s[:, 0, :], label1d_t1s[:, 1, :], vmasks, configs.loss.t1_temperature)
        loss_student_t1 = torch.mean(calculate_adapt_cof(label1d_t1s, label1ds) * loss_student_t1)
        loss += loss_student_t1 * configs.loss.t1_cof

        label1d_t2s = data['label1d_t2s']
        loss_student_t2 = lossfun_softloc(slogits, elogits, label1d_t2s[:, 0, :], label1d_t2s[:, 1, :], vmasks, configs.loss.t2_temperature)
        loss_student_t2 = torch.mean(calculate_adapt_cof(label1d_t2s, label1ds) * loss_student_t2)
        loss += loss_student_t2 * configs.loss.t2_cof

    return loss, output


def infer_MultiTeacher(output, configs):
    from utils.engine import infer_basic

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res