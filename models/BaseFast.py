import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


from models.layers import  VisualProjection, PositionalEmbedding, Conv1D, SeqPANPredictor
from models.layers import  WordEmbedding, FeatureEncoder

# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, bias=False):
#         super(SeparableConv2d, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
#                                 groups=in_channels, bias=bias, padding="same")
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 
#                                 kernel_size=1, bias=bias,  padding="same")
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

# class FeatureEncoder(nn.Module):
#     def __init__(self, dim, max_pos_len, kernel_size=7, num_layers=4, droprate=0.0):
#         super(FeatureEncoder, self).__init__()
#         # self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
#         self.conv_block = nn.ModuleList([ 
#             SeparableConv2d(in_channels=dim, out_channels=dim, kernel_size=(1, kernel_size),  bias=True)
#             for _ in range(num_layers)])
#         self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(p=droprate)

#         # self.conv_block = SeparableConv2d(in_channels=dim, out_channels=dim, kernel_size=(1, kernel_size),  bias=True)
#         # self.conv2d1 = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size),  bias=True, padding="same")
#         # self.conv2d2 = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1),  bias=True, padding="same")
#         # self.conv2d3 = nn.Conv2d(dim, dim, kernel_size=(kernel_size, kernel_size),  bias=True, padding="same")
#     def forward(self, x):
#         output = x  # (batch_size, seq_len, dim)
#         for idx, conv_layer in enumerate(self.conv_block):
#             residual = output
#             output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
#             output = output.transpose(1, 2).unsqueeze(2)  # (batch_size, dim, seq_len)
#             output = conv_layer(output)
#             output = self.dropout(output)
#             output = output.squeeze(2).transpose(1, 2)  # (batch_size, dim, seq_len)
#             output = output + residual
#         # output = self.conv_block(output)
#         return output


class BaseFast(nn.Module):
    def __init__(self, configs, word_vectors):
        super(BaseFast, self).__init__()
        self.configs = configs
        D = configs.model.dim
        self.num_props = 8
        droprate = configs.model.droprate
        self.vocab_size = configs.num_words
        max_pos_len = self.configs.model.vlen

        # self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=D,
        #                                word_dim=configs.model.word_dim, 
        #                                char_dim=configs.model.char_dim, 
        #                                word_vectors=word_vectors,
        #                                droprate=droprate)

        self.word_emb = WordEmbedding(configs.num_words, configs.model.word_dim, 0, word_vectors=word_vectors)

        self.text_conv1d = Conv1D(in_dim=configs.model.word_dim, out_dim=D)
        self.text_layer_norm = nn.LayerNorm(D, eps=1e-6)
        
        self.text_encoder = FeatureEncoder(dim=D, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=D, droprate=droprate)
        self.video_encoder = FeatureEncoder(dim=D, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.predictor = SeqPANPredictor(configs)

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        words_feat = self.word_emb(word_ids)
        tfeat = self.text_conv1d(words_feat)
        tfeat = self.text_layer_norm(tfeat)
        tfeat = self.text_encoder(tfeat)

        vfeat = self.video_affine(vfeat_in)
        vfeat = self.video_encoder(vfeat)

        f_tfeat = torch.max(tfeat, dim=1)[0]
        f_fusion = vfeat * f_tfeat.unsqueeze(1)
        slogits, elogits = self.predictor(f_fusion, vmask)


        res = {
            "slogits": slogits,
            "elogits": elogits,
            "vmask" : vmask
        }

        return res


def train_engine_BaseFast(model, data, configs):
    from models.loss import lossfun_loc
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    slogits = output["slogits"]
    elogits = output["elogits"]

    dist_idxs =  data['dist_idxs']
    loc_loss = lossfun_loc(slogits, elogits, dist_idxs[:, 0, :], dist_idxs[:, 1, :], data['vmasks'])
    loss =loc_loss 
    return loss, output


def infer_BaseFast(output, configs):
    from utils.engine import infer_basic

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    sfrac, efrac = infer_basic(start_logits, end_logits, vmask)

    res = np.stack([sfrac, efrac]).T
    return res