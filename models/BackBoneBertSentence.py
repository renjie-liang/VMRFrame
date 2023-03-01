import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock

class BackBoneBertSentence(nn.Module):
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

        self.text_affine = VisualProjection(visual_dim=768, dim=dim,droprate=droprate)
        self.tfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim,droprate=droprate)
        self.vfeat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,max_pos_len=max_pos_len, droprate=droprate)


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


    def forward(self, sentence_embeddings, tmask, vfeat_in, vmask):
        torch.cuda.synchronize()
        start = time.time()
        B = vmask.shape[0]
        # tfeat = self.text_encoder(sentence_embeddings)
        tfeat = self.text_affine(sentence_embeddings)
        tfeat = self.tfeat_encoder(tfeat)

        vfeat = self.video_affine(vfeat_in)
        vfeat = self.vfeat_encoder(vfeat)


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
from sentence_transformers import SentenceTransformer

class BackBoneBertSentenceDataset(BaseDataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super().__init__(dataset, video_features, configs, loadertype)
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def __getitem__(self, index):
        res = BaseDataset.__getitem__(self, index)
        sentence = res["record"]["sentence"]
        sentence_embedding = self.model.encode(sentence, show_progress_bar=False)
        res["sentence_embedding"] = torch.from_numpy(sentence_embedding)
        return res
    
class BackBoneBertSentenceCollate(BaseCollate):
    def __call__(self, datas):
        res, records = super().__call__(datas)
        sentence_embeddings = []
        for d in datas:
            sentence_embeddings.append(d["sentence_embedding"].unsqueeze(0))
        res["sentence_embeddings"] = torch.stack(sentence_embeddings)
        res["tmasks"] = torch.ones([len(datas), 1])
        return res, records



import time
from models.loss import lossfun_loc, lossfun_match
def train_engine_BackBoneBertSentence(model, data, configs, runmode):
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['sentence_embeddings'], data['tmasks'], data['vfeats'], data['vmasks'])
    # location loss
    slogits = output["slogits"]
    elogits = output["elogits"]
    label1ds =  data['label1ds']
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], data['vmasks'])
    m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  data['vmasks'])

    loss =loc_loss + m_loss
    return loss, output


def infer_BackBoneBertSentence(output, configs):
    from utils.engine import infer_basic

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    res = infer_basic(start_logits, end_logits, vmask)
    return res