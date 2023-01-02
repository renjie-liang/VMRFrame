import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


from models.layers import  VisualProjection, PositionalEmbedding, Conv1D, SeqPANPredictor
from models.layers import  WordEmbedding, FeatureEncoder
from utils.utils import generate_2dmask


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
        droprate = configs.model.droprate
        self.vocab_size = configs.num_words
        max_pos_len = self.configs.model.vlen

        self.logit2D_mask = generate_2dmask(max_pos_len).to(configs.device)
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

        logit2Ds = torch.matmul(slogits.unsqueeze(2), elogits.unsqueeze(1)) * self.logit2D_mask
        
        res = {
            "slogits": slogits,
            "elogits": elogits,
            "vmask" : vmask,
            "logit2D_mask" : self.logit2D_mask,
            "logit2Ds": logit2Ds,
        }
        return res



# ---------------------------------
def collate_fn_BaseFast(datas):
    from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
    from utils.utils import convert_length_to_mask

    records, se_times, se_fracs = [], [], []
    vfeats, words_ids, chars_ids = [], [], []
    label1ds, label2ds = [], []
    max_vlen = datas[0]["max_vlen"]
    for d in datas:
        records.append(d["record"])
        vfeats.append(d["vfeat"])
        words_ids.append(d["words_id"])
        label1ds.append(d["label1d"])
        label2ds.append(d["label2d"])
        se_times.append(d["se_time"])
        se_fracs.append(d["se_frac"])
        chars_ids.append(d["chars_id"])

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
    label1ds = torch.stack(label1ds)
    label2ds = torch.stack(label2ds)
    
    se_times = torch.as_tensor(se_times, dtype=torch.float)
    se_fracs = torch.as_tensor(se_fracs, dtype=torch.float)

    res = {'words_ids': words_ids,
            'char_ids': chars_ids,
            'tmasks': tmasks,

            'vfeats': vfeats,
            'vmasks': vmasks,

            # labels
            'label1ds': label1ds,
            'label2ds': label2ds,

            # evaluate
            'se_times': se_times,
            'se_fracs': se_fracs,
            # 'vname': vname,
            # 'sentence': sentence
            }

    return res, records

def lossfun_loc2d(scores2d, labels2d, mask2d):
    def scale(iou, min_iou, max_iou):
        return (iou - min_iou) / (max_iou - min_iou)

    labels2d = scale(labels2d, 0.5, 1.0).clamp(0, 1)
    loss_loc2d = F.binary_cross_entropy_with_logits(
        scores2d.squeeze().masked_select(mask2d),
        labels2d.masked_select(mask2d)
    )
    return loss_loc2d

def train_engine_BaseFast(model, data, configs):
    from models.loss import lossfun_loc
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    slogits = output["slogits"]
    elogits = output["elogits"]

    label1ds =  data['label1ds']
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], data['vmasks'])
    loc2d_loss = lossfun_loc2d(output["logit2Ds"], data["label2ds"], output['logit2D_mask'])


    loss = loc2d_loss
    return loss, output


# def infer_BaseFast(output, configs):
#     from utils.engine import infer_basic

#     start_logits = output["slogits"]
#     end_logits = output["elogits"]
#     vmask = output["vmask"]
#     sfrac, efrac = infer_basic(start_logits, end_logits, vmask)

#     res = np.stack([sfrac, efrac]).T
#     return res


def infer_BaseFast(output, configs):
    vmask = output["vmask"]
 
    outer = torch.triu(output["logit2Ds"], diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res