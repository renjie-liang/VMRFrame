import torch
import torch.nn as nn
from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock
import torch.nn.functional as F
class SeqPAN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(SeqPAN, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate

        max_pos_len = self.configs.max_pos_len
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)
                                       
        self.video_affine = VisualProjection(visual_dim=configs.model.video_feature_dim, dim=dim,
                                             droprate=droprate)
        self.feat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4,
                                              max_pos_len=max_pos_len, droprate=droprate)


        self.dual_attention_block_1 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                        droprate=droprate, use_bias=True, activation=None)
        self.dual_attention_block_2 = DualAttentionBlock(configs=configs, dim=dim, num_heads=configs.model.num_heads, 
                                                    droprate=droprate, use_bias=True, activation=None)



        # self.tfeat_encoder = FeatureEncoder(dim=dim, num_heads=configs.model.num_heads, kernel_size=7, num_layers=4,
        #                                       max_pos_len=max_pos_len, droprate=droprate)

        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)
        self.label_embs = torch.empty(size=[dim, 4], dtype=torch.float32)
        self.label_embs = torch.nn.init.orthogonal_(self.label_embs.data)

        self.predictor = SeqPANPredictor(configs)

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        B = vmask.shape[0]

        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)

        vfeat = self.feat_encoder(vfeat)
        tfeat = self.feat_encoder(tfeat)

        vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        vfeat = vfeat_
        tfeat = tfeat_

        vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)

        vfeat = vfeat_
        tfeat = tfeat_

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)


        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)

        self.label_embs = self.label_embs.to(vmask.device)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))

        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        start_logits, end_logits = self.predictor(fuse_feat, vmask)
        return start_logits, end_logits, match_probs, self.label_embs