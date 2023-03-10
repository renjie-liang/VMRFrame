import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock
from models.ActionFormerlib.meta_archs import ConvTransformerBackbone

class BackBoneActionFormer(nn.Module):
    def __init__(self, configs, word_vectors):
        super().__init__()
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
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim, droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,max_pos_len=max_pos_len, droprate=droprate)
        self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)


        # self.actionformer_BackBoneActionFormer = 
        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)

        self.predictor = SeqPANPredictor(configs)


        self.backbone = ConvTransformerBackbone(
                        **{
                            'n_in' : 128,
                            'n_embd' : 128,
                            'n_head': 4,
                            'n_embd_ks': 3,
                            'max_len': 64,
                            'arch' : [2, 2, 3],
                            'mha_win_size': [5, 5, 5, -1],
                            'scale_factor' : 2,
                            'with_ln' : True,
                            'attn_pdrop' : 0.0,
                            'proj_pdrop' : 0.0,
                            'path_pdrop' : 0.1,
                            'use_abs_pe' : True,
                            'use_rel_pe' : False
                        }
                    )





    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        torch.cuda.synchronize()
        start = time.time()
        
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

        fuse_feats, vmasks = self.backbone(fuse_feat.permute(0, 2, 1), vmask.unsqueeze(1))
        fuse_feat, vmask = fuse_feats[0].permute(0, 2, 1), vmasks[0].squeeze(1)
        slogits, elogits = self.predictor(fuse_feat, vmask)


        torch.cuda.synchronize()
        end = time.time()
        consume_time = end - start
        return {    "slogits": slogits,
                    "elogits": elogits,
                    "vmask" : vmask,
                    # "match_score" : match_score,
                    # "label_embs" : self.label_embs,
                    "consume_time": consume_time,
                    }

from utils.BaseDataset import BaseDataset, BaseCollate
class BackBoneActionFormerDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        return res
    
class BackBoneActionFormerCollate(BaseCollate):
    def __call__(self, datas):
        return super().__call__(datas)
    

import time
def train_engine_BackBoneActionFormer(model, data, configs, runmode):
    from models.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}

    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])
    # location loss
    slogits = output["slogits"]
    elogits = output["elogits"]
    label1ds =  data['label1ds']
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], data['vmasks'])
    loss =loc_loss 
    return loss, output

def infer_BackBoneActionFormer(output, configs):
    from utils.engine import infer_basic
    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res