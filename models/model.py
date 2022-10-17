import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock, WordEmbedding
from models.cpl_utils import TransformerDecoder, _generate_mask, SinusoidalPositionalEmbedding
class SeqPAN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(SeqPAN, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        self.num_props = 8
        droprate = configs.model.droprate
        self.vocab_size = configs.num_words

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

        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        
        self.predictor = SeqPANPredictor(configs)

        self.decoder1 = TransformerDecoder(num_layers=2, d_model=dim, num_heads=4, dropout=0.1)
        self.decoder2 = TransformerDecoder(num_layers=2, d_model=dim, num_heads=4, dropout=0.1)
        self.word_emb = WordEmbedding(configs.num_words, configs.model.word_dim, 0.0, word_vectors=word_vectors)
        self.word_fc = nn.Linear(configs.model.word_dim, dim)
        self.start_vec = nn.Parameter(torch.zeros(configs.model.word_dim).float(), requires_grad=True)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(dim, 0, 20)
        self.conv1d_cw = nn.Conv1d(in_channels=max_pos_len, out_channels=1, kernel_size=1)

        self.fc_gauss = nn.Linear(dim, self.num_props*2)
        self.fc_comp = nn.Linear(dim, self.vocab_size)

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        B, L, D = vfeat_in.shape
        P = self.num_props
        # tfeat = self.text_encoder(word_ids, char_ids)
        vfeat_tmp = self.video_affine(vfeat_in)

        #### CPL
        words_feat = self.word_emb(word_ids)
        tmp = torch.zeros([B, 1, words_feat.shape[-1]]).cuda()
        words_feat = torch.concat([tmp, words_feat], dim=1)
        words_feat[:, 0] = self.start_vec
        words_feat = F.dropout(words_feat, 0.1, self.training)
        tfeat_long = self.word_fc(words_feat)
        tmask_long = _generate_mask(words_feat, tmask.sum(dim=1).long() + 1)

        # words_pos = self.word_pos_encoder(words_feat)
        # tfeat = tfeat + words_pos

        # SeqPAN
        tfeat = tfeat_long[:, 1:]
        tmask = tmask_long[:, 1:]

        vfeat = self.feat_encoder(vfeat_tmp)
        tfeat = self.feat_encoder(tfeat)

        tfeat_, _ = self.decoder1(vfeat, vmask, tfeat, tmask)
        vfeat_, _ = self.decoder2(tfeat, tmask, vfeat, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        tfeat_, _ = self.decoder1(vfeat, vmask, tfeat, tmask)
        vfeat_, _ = self.decoder2(tfeat, tmask, vfeat, vmask)
        vfeat, tfeat = vfeat_, tfeat_

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)

        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        start_logits, end_logits = self.predictor(fuse_feat, vmask)


        ### CPL
        props_len = L

        weakly_feat = self.conv1d_cw(vfeat_tmp).squeeze()
        gauss_param = torch.sigmoid(self.fc_gauss(weakly_feat)).view(B*P, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]
        
        vfeat_props = torch.repeat_interleave(vfeat_tmp, P, dim=0)
        vmask_props = torch.repeat_interleave(vmask, P, dim=0)
        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width, vmask_props)
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]

        tmask_props = torch.repeat_interleave(tmask_long[:, :-1], P, dim=0)
        tfeat_props = torch.repeat_interleave(tfeat_long[:, :-1], P, dim=0)

        enc_out, _ = self.decoder1(None, None, vfeat_props, vmask_props, tgt_gauss_weight=pos_weight)
        out, weight = self.decoder2(enc_out, vmask_props, tfeat_props, tmask_props, src_gauss_weight=pos_weight)
        words_logit = self.fc_comp(out)

        res = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "match_score": match_score,
            "label_embs": self.label_embs,
            'word_ids': word_ids,
            'words_mask': tmask_long[:, :-1],
            'words_logit': words_logit,
            'gauss_weight': gauss_weight,
            'width': gauss_width,
            'center': gauss_center,
        }

        return res




    def generate_gauss_weight(self, props_len, center, width, vmask):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        center = center * ( vmask.sum(dim=1) / vmask.shape[1])
        center = center.unsqueeze(-1)
        width = width * ( vmask.sum(dim=1) / vmask.shape[1])
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]



# from models.layers_vsl import *
# class VSLNet(nn.Module):
#     def __init__(self, configs, word_vectors):
#         super(VSLNet, self).__init__()
#         self.configs = configs
#         self.embedding_net = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=configs.model.dim,
#                                        word_dim=configs.model.word_dim, char_dim=configs.model.char_dim, 
#                                        word_vectors=word_vectors,
#                                        drop_rate=configs.droprate)



#         self.video_affine = VisualProjection(visual_dim=configs.model.video_feature_dim, dim=configs.model.dim,
#                                              drop_rate=configs.droprate)
#         self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.model.num_heads, kernel_size=7, num_layers=4,
#                                               max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)

                                              
#         self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
#         self.cq_concat = CQConcatenate(dim=configs.dim)
#         self.highlight_layer = HighLightLayer(dim=configs.dim)
#         self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.model.num_heads, drop_rate=configs.drop_rate,
#                                               max_pos_len=configs.max_pos_len, predictor=configs.predictor)
#         self.init_parameters()

#     def init_parameters(self):
#         def init_weights(m):
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     torch.nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.LSTM):
#                 m.reset_parameters()
#         self.apply(init_weights)

#     def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):
#         video_features = self.video_affine(video_features)
#         query_features = self.embedding_net(word_ids, char_ids)
#         video_features = self.feature_encoder(video_features, mask=v_mask)
#         query_features = self.feature_encoder(query_features, mask=q_mask)
#         features = self.cq_attention(video_features, query_features, v_mask, q_mask)
#         features = self.cq_concat(features, query_features, q_mask)
#         h_score = self.highlight_layer(features, v_mask)
#         # features = features * h_score.unsqueeze(2)
#         start_logits, end_logits = self.predictor(features, mask=v_mask)
#         return start_logits, end_logits


    

        # vfeat_ = self.dual_attention_block_1(vfeat, tfeat, vmask, tmask)
        # tfeat_ = self.dual_attention_block_1(tfeat, vfeat, tmask, vmask)
        # vfeat, tfeat = vfeat_, tfeat_

        # vfeat_ = self.dual_attention_block_2(vfeat, tfeat, vmask, tmask)
        # tfeat_ = self.dual_attention_block_2(tfeat, vfeat, tmask, vmask)
        # vfeat, tfeat = vfeat_, tfeat_
