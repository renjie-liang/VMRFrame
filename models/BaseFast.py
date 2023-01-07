import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


from models.layers import  VisualProjection, PositionalEmbedding, Conv1D, SeqPANPredictor
from models.layers import  Embedding, WordEmbedding#, FeatureEncoder, 
from utils.utils import generate_2dmask


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding="same")
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias,  padding="same")
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class FeatureEncoder(nn.Module):
    def __init__(self, dim, max_pos_len, kernel_size=7, num_layers=4, droprate=0.0):
        super(FeatureEncoder, self).__init__()
        # self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim)
        self.conv_block = nn.ModuleList([ 
            SeparableConv2d(in_channels=dim, out_channels=dim, kernel_size=(1, kernel_size),  bias=True)
            for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.conv_block):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2).unsqueeze(2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.squeeze(2).transpose(1, 2)  # (batch_size, dim, seq_len)
            output = output + residual
        # output = self.conv_block(output)
        return output


def load_commonsense_emb(attri_input_path, commonsense_path):
    import pickle
    attribute_input_emb = pickle.load(open(attri_input_path, 'rb'))
    com_dict = pickle.load(open(commonsense_path, 'rb'))
    com_vectors = []
    for k in com_dict.keys():
        com_vectors.append(com_dict[k])
    com_vectors = np.array(com_vectors)
    attribute_input_emb = np.concatenate([attribute_input_emb, com_vectors], 0)
    attribute_input_emb = torch.from_numpy(attribute_input_emb)
    return attribute_input_emb


from models.CCA import C_GCN
class BaseFast(nn.Module):
    def __init__(self, configs, word_vectors):
        super(BaseFast, self).__init__()
        self.configs = configs
        D = configs.model.dim
        droprate = configs.model.droprate
        self.vocab_size = configs.num_words
        max_pos_len = self.configs.model.vlen

        self.logit2D_mask = generate_2dmask(max_pos_len).to(configs.device)

        # self.word_emb = WordEmbedding(configs.num_words, configs.model.word_dim, 0, word_vectors=word_vectors)
        self.wordchat_emb = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=D,
                                       word_dim=configs.model.word_dim, 
                                       char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors,
                                       droprate=droprate)

        self.text_conv1d = Conv1D(in_dim=configs.model.word_dim, out_dim=D)
        self.text_layer_norm = nn.LayerNorm(D, eps=1e-6)
        
        self.text_encoder = FeatureEncoder(dim=D, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=D, droprate=droprate)
        self.video_encoder = FeatureEncoder(dim=D, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.predictor = SeqPANPredictor(configs)

        # # CCA
        # self.concept_input_embs = load_commonsense_emb(configs.paths.attri_input_path, configs.paths.commonsense_path).to(configs.device)
        # self.C_GCN = C_GCN(3152, in_channel=300, t=0.3, embed_size=1024, 
        #                     adj_file="/storage/rjliang/4_FastVMR/CCA/acnet_concept/acnet_concept_adj.pkl",
        #                     norm_func='sigmoid', 
        #                     num_path='/storage/rjliang/4_FastVMR/CCA/acnet_concept/acnet_dict.pkl', 
        #                     com_path='/storage/rjliang/4_FastVMR/CCA/acnet_concept/acnet_com_graph.pkl')
        # self.V_TransformerLayer = nn.TransformerEncoderLayer(3248, 8)


    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        # CCA
        # B, L, D = vfeat_in.shape
        # concept_input = self.concept_input_embs[None, :, :].repeat(B, 1, 1)
        # concept_basis = self.C_GCN(concept_input)
        # vfeat = torch.cat([vfeat_in.permute(0, 2, 1), concept_basis.unsqueeze(0).repeat(vfeat_in.size(0), 1, 1).permute(0, 2, 1)], dim=2)
        # vfeat = self.V_TransformerLayer(vfeat)[:, :, :96].permute(0, 2, 1)

        # words_feat = self.word_emb(word_ids)
        # tfeat = self.text_conv1d(words_feat)
        # tfeat = self.text_layer_norm(tfeat)
        tfeat = self.wordchat_emb(word_ids, char_ids)
        tfeat = self.text_encoder(tfeat)

        vfeat = self.video_affine(vfeat_in)
        vfeat = self.video_encoder(vfeat)

        # tfeat = tfeat.masked_select(tmask)
        f_tfeat = torch.max(tfeat * tmask[1, 1, None], dim=1)[0]
        f_fusion = vfeat * f_tfeat.unsqueeze(1)
        slogits, elogits = self.predictor(f_fusion, vmask)

        # slogits = torch.sigmoid(slogits)
        # elogits = torch.sigmoid(elogits)
        logit2Ds = torch.matmul(slogits.unsqueeze(2), elogits.unsqueeze(1)) * self.logit2D_mask
        
        res = {
            "tfeat": tfeat,
            "vfeat": vfeat,
            
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
    label1ds, label2ds, label1d_model1s, NER_labels = [], [], [], []
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
        label1d_model1s.append(d["label1d_model1"])
        NER_labels.append(d["NER_label"])
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
    label1d_model1s = torch.stack(label1d_model1s)
    NER_labels = torch.stack(NER_labels)
    
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
            "label1d_model1s": label1d_model1s,
            "NER_labels": NER_labels,

            # evaluate
            'se_times': se_times,
            'se_fracs': se_fracs,

            # 'vname': vname,
            # 'sentence': sentence
            }

    return res, records


def lossfun_softloc(slogits, elogits, s_labels, e_labels, vmask, temperature):
    from models.layers import mask_logits
    slogits = mask_logits(slogits, vmask)
    elogits = mask_logits(elogits, vmask)
    s_labels = mask_logits(s_labels, vmask)
    e_labels = mask_logits(e_labels, vmask)
    
    slogits = F.softmax(F.normalize(slogits, p=2, dim=1) / temperature, dim=-1) 
    elogits = F.softmax(F.normalize(elogits, p=2, dim=1) / temperature, dim=-1) 
    s_labels = F.softmax(F.normalize(s_labels, p=2, dim=1) / temperature, dim=-1) 
    e_labels = F.softmax(F.normalize(e_labels, p=2, dim=1) / temperature, dim=-1) 


    # sloss = F.cross_entropy(slogits, s_labels, reduce="batchmean")
    # eloss = F.cross_entropy(elogits, e_labels, reduce="batchmean")

    sloss = F.kl_div(slogits.log(), s_labels, reduction='sum')
    eloss = F.kl_div(elogits.log(), e_labels, reduction='sum')

    return sloss + eloss


def lossfun_aligment(tfeat, vfeat, tmask, vmask, inner_label):
    tfeat = tfeat.sum(1) / tmask.sum(1).unsqueeze(1)
    tfeat = F.normalize(tfeat, p=2, dim=1)  # B, channels
    frame_weights = inner_label / vmask.sum(1, keepdim=True)

    vfeat = vfeat * frame_weights.unsqueeze(2)
    vfeat = vfeat.sum(1)
    vfeat = F.normalize(vfeat, p=2, dim=1)
    video_sim = torch.matmul(vfeat, vfeat.T)
    video_sim = torch.softmax(video_sim, dim=-1)
    query_sim = torch.matmul(tfeat, tfeat.T)
    query_sim = torch.softmax(query_sim, dim=-1)
    kl_loss = (F.kl_div(query_sim.log(), video_sim, reduction='sum') +
                F.kl_div(video_sim.log(), query_sim, reduction='sum')) / 2
    return kl_loss



def train_engine_BaseFast(model, data, configs):
    from models.loss import lossfun_loc, lossfun_loc2d
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    slogits = output["slogits"]
    elogits = output["elogits"]

    label1ds =  data['label1ds']
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], data['vmasks'])
    
    # label1d_model1s = data["label1d_model1s"]
    # softloc_loss = lossfun_softloc(slogits, elogits, label1d_model1s[:,0,:], label1d_model1s[:, 1, :], data['vmasks'], 3)
    # # loc2d_loss = lossfun_loc2d(output["logit2Ds"], data["label2ds"], output['logit2D_mask'])

    NER_labels = data['NER_labels']
    NER_labels[NER_labels != 0] = 1
    align_loss = lossfun_aligment(output["tfeat"], output["vfeat"], data['tmasks'], data['vmasks'], NER_labels)
    loss = loc_loss + 1.0 * align_loss# + 1.0 *softloc_loss# + loc2d_loss # + 0.5*softloc_loss + loc2d_loss
    return loss, output


def infer_BaseFast(output, configs):
    from utils.engine import infer_basic

    start_logits = output["slogits"]
    end_logits = output["elogits"]
    vmask = output["vmask"]
    sfrac, efrac = infer_basic(start_logits, end_logits, vmask)

    res = np.stack([sfrac, efrac]).T
    return res


# def infer_BaseFast(output, configs):
#     vmask = output["vmask"]
 
#     outer = torch.triu(output["logit2Ds"], diagonal=0)
#     _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
#     _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
#     sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
#     efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
#     res = np.stack([sfrac, efrac]).T
#     return res