import torch
import torch.nn as nn
from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
import torch.nn.functional as F

class SeqPAN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(SeqPAN, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate

        max_pos_len = self.configs.max_pos_len

        # self.dropout = nn.Dropout(droprate)
        # self.linear_1= Conv1D(in_dim=1024, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.video_affine = VisualProjection(visual_dim=configs.model.video_feature_dim, dim=dim,
                                             droprate=droprate)
        self.feature_encoder = FeatureEncoder(dim=dim, num_heads=configs.model.num_heads, kernel_size=7, num_layers=4,
                                              max_pos_len=max_pos_len, droprate=droprate)

        self.cq_attention = CQAttention(dim=dim, droprate=droprate)
        self.cq_concat = CQConcatenate(dim=dim)

        self.conv1d = Conv1D(in_dim=dim, out_dim=4)

        self.label_embs = torch.empty(size=[dim, 4], dtype=torch.float32)
        self.label_embs = torch.nn.init.orthogonal_(self.label_embs.data)

        self.predictor = SeqPANPredictor(configs)

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        B = vmask.shape[0]

        vfeat = self.video_affine(vfeat_in)
        vfeat = self.layer_norm(vfeat)
        vfeat = self.feature_encoder(vfeat, mask=vmask)

        tfeat = self.text_encoder(word_ids, char_ids)
        tfeat = self.layer_norm(tfeat)
        tfeat = self.feature_encoder(tfeat, mask=tmask)

        t2v_feat = self.cq_attention(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.cq_attention(tfeat, vfeat, tmask, vmask)
        
        fuse_feat = self.cq_concat(t2v_feat, v2t_feat, tmask)

        match_logits = self.conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        match_probs =torch.log(match_score)

        self.label_embs = self.label_embs.to(vmask.device)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))


        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)

        start_logits, end_logits = self.predictor(fuse_feat, vmask)

        return start_logits, end_logits, match_probs, self.label_embs
    #     query_features = self.embedding_net(word_ids, char_ids)
    #     video_features = self.feature_encoder(video_features, mask=v_mask)
    #     query_features = self.feature_encoder(query_features, mask=q_mask)
    #     features = self.cq_attention(video_features, query_features, v_mask, q_mask)
    #     features = self.cq_concat(features, query_features, q_mask)
    #     h_score = self.highlight_layer(features, v_mask)
    #     features = features * h_score.unsqueeze(2)
    #     start_logits, end_logits = self.predictor(features, mask=v_mask)
    #     return h_score, start_logits, end_logits,

    # def extract_index(self, start_logits, end_logits):
    #     return self.predictor.extract_index(start_logits=start_logit
    # 
    #   # def compute_highlight_loss(self, scores, labels, mask):
    #     return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    # def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
    #     return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
    #                                                      start_labels=start_labels, end_labels=end_labels)
