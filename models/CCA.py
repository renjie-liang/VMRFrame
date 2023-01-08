import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.nn import Parameter
import math
import pickle


def gen_A_concept(num_classes, t, adj_file, num_path=None, com_path=None):
    import pickle
    result = pickle.load(open(adj_file, 'rb')).numpy()
    for idx in range(result.shape[0]):
        result[idx][idx] = 0

    _nums = get_num(num_path)

    _A_adj = {}
    
    _adj_all = result
    _adj_all = _adj_all / _nums

    _adj_all = rescale_adj_matrix(_adj_all)
    _adj_all[_adj_all < t] = 0
    _adj_all[_adj_all >= t] = 1 
    _adj_all = generate_com_weight(_adj_all, com_path)
    _adj_all = _adj_all * 0.25 / (_adj_all.sum(0, keepdims=True) + 1e-6)
    _adj_all = _adj_all + np.identity(num_classes, np.int)  # identity square matrix
    _A_adj['adj_all'] = _adj_all

    return _A_adj


def rescale_adj_matrix(adj_mat, t=5, p=0.02):

    adj_mat_smooth = np.power(t, adj_mat - p) - np.power(t,  -p)
    return adj_mat_smooth


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def get_num(path=None):
    concept_dict = pickle.load(open(path, 'rb'))
    num = len(concept_dict)
    _num = np.zeros([num, 1], dtype=np.int32)
    key_list = list(concept_dict.keys())
    for idx in range(len(key_list)):
        _num[idx][0] = concept_dict[key_list[idx]]
    return _num

def generate_com_weight(_adj_all, com_path):

    com_weight = pickle.load(open(com_path, 'rb'))
    train_length = _adj_all.shape[0]
    com_length = com_weight.shape[0]
    all_length = train_length + com_length
    _adj = np.zeros([all_length, all_length], dtype=np.int32)
    _adj[:train_length, :train_length] = _adj_all
    _adj[train_length:, :] = com_weight
    _adj[:, train_length:] = np.transpose(com_weight)
    return _adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, which shared the weight between two separate graphs
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj['adj_all'], support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class C_GCN(nn.Module):

    def __init__(self, num_classes, in_channel=300, t=0, embed_size=None, adj_file=None, norm_func='sigmoid', num_path=None, com_path=None):
        super(C_GCN, self).__init__()

        self.num_classes = num_classes
        self.gc1 = GraphConvolution(in_channel, embed_size // 2)
        self.gc2 = GraphConvolution(embed_size // 2,  embed_size)
        self.relu = nn.LeakyReLU(0.2)

        # concept correlation mat generation
        _adj = gen_A_concept(num_classes, t, adj_file, num_path=num_path, com_path=com_path)

        self.adj_all = Parameter(torch.from_numpy(_adj['adj_all']).float())

        self.norm_func = norm_func
        self.softmax = nn.Softmax(dim=1)
        self.joint_att_emb = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.init_weights()

    def init_weights(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_size + self.embed_size)
        self.joint_att_emb.weight.data.uniform_(-r, r)
        self.joint_att_emb.bias.data.fill_(0)


    def forward(self, inp):

        inp = inp[0]

        adj_all = gen_adj(self.adj_all).detach()

        adj = {}

        adj['adj_all'] = adj_all

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        concept_feature = x
        concept_feature = l2norm(concept_feature)

        return concept_feature


def l2norm(input, axit=-1):
    norm = torch.norm(input, p=2, dim=-1, keepdim=True) + 1e-12
    output = torch.div(input, norm)
    return output

# ---------- layers ------
class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).relu()
        feat = self.pool(x)
        return feat

def build_featpool(cfg):
    input_size = cfg.MODEL.CCA.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.CCA.FEATPOOL.KERNEL_SIZE
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.CCA.NUM_CLIPS
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)


import torch
from torch import nn

class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                # fill a diagonal line 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape # (32, 512, 128)
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x) # (32, 512, 127)
            map2d[:, :, i, j] = x
        return map2d

def build_feat2d(cfg):
    pooling_counts = cfg.MODEL.CCA.FEAT2D.POOLING_COUNTS
    num_clips = cfg.MODEL.CCA.NUM_CLIPS
    return SparseMaxPool(pooling_counts, num_clips)




import torch
from torch import nn
from torch.functional import F

def mask2weight(mask2d, mask_kernel, padding=0):
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight

class SimPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d, feat_hidden_size, query_input_size, query_hidden_size,
                 bidirectional, num_layers):
        super(SimPredictor, self).__init__()

        if bidirectional:
            query_hidden_size //= 2
        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True
        )
        if bidirectional:
            query_hidden_size *= 2
        self.fc_full = nn.Linear(query_hidden_size, feat_hidden_size)

        self.conv = nn.Conv2d(hidden_size, feat_hidden_size, 5, padding=2)
        self.bn = nn.BatchNorm2d(feat_hidden_size)
        self.conv1 = nn.Conv2d(feat_hidden_size, feat_hidden_size, 3, padding=1)

    def encode_query(self, queries, wordlens):

        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]
        queries_start = queries[range(queries.size(0)),0]
        queries_end = queries[range(queries.size(0)), wordlens.long() - 1]
        full_queries = (queries_start + queries_end)/2

        return self.fc_full(full_queries)

    def forward(self, batch_queries, wordlens, map2d):

        queries = self.encode_query(batch_queries, wordlens)
        map2d = self.conv(map2d)
        map2d = F.tanh(self.bn(map2d))
        map2d = self.conv1(map2d)
        return map2d, queries


def build_simpredictor(cfg, mask2d):
    input_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    hidden_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.CCA.PREDICTOR.KERNEL_SIZE
    num_stack_layers = cfg.MODEL.CCA.PREDICTOR.NUM_STACK_LAYERS
    feat_hidden_size = cfg.MODEL.CCA.FEATPOOL.HIDDEN_SIZE
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.CCA.INTEGRATOR.QUERY_HIDDEN_SIZE
    bidirectional = cfg.MODEL.CCA.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.CCA.INTEGRATOR.LSTM.NUM_LAYERS

    return SimPredictor(
        input_size, hidden_size, kernel_size, num_stack_layers, mask2d, feat_hidden_size, query_input_size, query_hidden_size,
        bidirectional, num_layers
    )


class FuseAttention(nn.Module):
    def __init__(self, hidden_dim, concept_dim, norm=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.query = nn.Linear(self.hidden_dim, self.concept_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm = norm
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, feat, concept):

        query = self.query(feat)
        key = self.key(concept)
        value = self.value(concept)

        attention_scores = torch.matmul(query, key.transpose(1, 0))
        attention_scores = nn.Softmax(dim=1)(attention_scores * 10)
        attention_scores = self.dropout(attention_scores)

        out = torch.matmul(attention_scores, value)

        if self.norm:
            out = l2norm(out + feat)

        return out




# ---------------------- MODEL ----------------
import torch
from torch import nn
from models.layers import  WordEmbedding

class CCA(nn.Module):
    def __init__(self, cfg, word_vectors):
        super(CCA, self).__init__()
        self.device = cfg.device

        self.word_emb = WordEmbedding(cfg.num_words, cfg.model.word_dim, 0, word_vectors=word_vectors)


        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.simpredictor = build_simpredictor(cfg, self.feat2d.mask2d)
        self.T_fuse_attn = FuseAttention(512, 512, True)
        self.C_GCN = C_GCN(cfg.num_attribute, in_channel=cfg.input_channel, t=0.3, embed_size=cfg.embed_size, adj_file=cfg.adj_file,
                                norm_func=cfg.norm_func_type, num_path=cfg.num_path, com_path=cfg.com_concept)
        
        self.v_t_param = nn.Parameter(torch.FloatTensor([0.5]))

        self.concept_dim = cfg.num_attribute
        # self.V_TransformerLayer = nn.TransformerEncoderLayer(cfg.MODEL.CCA.NUM_CLIPS, 8)
        self.V_TransformerLayer = nn.TransformerEncoderLayer(cfg.MODEL.CCA.NUM_CLIPS + self.concept_dim, 8)
        self.cut_dim = cfg.MODEL.CCA.NUM_CLIPS
    
    def forward(self, words_ids, tmask, vfeat_in, vmask, concept_input_embs):
        concept_basis = self.C_GCN(concept_input_embs)
        feats = self.featpool(vfeat_in)

        feats = torch.cat([feats, concept_basis.unsqueeze(0).repeat(feats.size(0), 1, 1).permute(0, 2, 1)], dim=2)
        feats = self.V_TransformerLayer(feats)[:, :, :self.cut_dim]
        map2d = self.feat2d(feats)

        tfeat = self.word_emb(words_ids)
        map2d_fused, queries = self.simpredictor(tfeat, tmask.sum(dim=1), map2d)

        queries_fused = self.T_fuse_attn(queries, concept_basis)
        # queries_fused = queries

        v2t_map2d = queries[:, :, None, None] * map2d_fused
        v2t_scores2d = torch.sum(F.normalize(v2t_map2d), dim=1).squeeze_()
        t2v_map2d = queries_fused[:, :, None, None] * map2d
        t2v_scores2d = torch.sum(F.normalize(t2v_map2d), dim=1).squeeze_()
        
        original_scores2d = self.v_t_param * v2t_scores2d + (1 - self.v_t_param) * t2v_scores2d
        
        res  = {"scores2d": original_scores2d,
                "vmask" : vmask}
        return res
        # if self.training:
        #     original_loss = self.ccaloss(original_scores2d, ious2d)
        #     return original_loss
        # else:
        #     return original_scores2d.sigmoid_() * self.feat2d.mask2d


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
attri_input_path = "/storage/rjliang/4_FastVMR/CCA/acnet_concept/acnet_concepts_glove_word2vec.pkl"
commonsense_path = "/storage/rjliang/4_FastVMR/CCA/acnet_concept/acnet_test_concepts.pkl"
concept_input_embs = load_commonsense_emb(attri_input_path, commonsense_path)

# ------------ run -------------
def collate_fn_CCA(datas):
    from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
    from utils.utils import convert_length_to_mask

    records, se_times, se_fracs = [], [], []
    vfeats, words_ids = [], []
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

    # process text
    words_ids, _ = pad_seq(words_ids)
    words_ids = torch.as_tensor(words_ids, dtype=torch.int64)
    tmasks = (torch.zeros_like(words_ids) != words_ids).float()
    
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

    B = len(datas)
    concept_inputs = concept_input_embs[None, :, :].repeat(B, 1, 1)

    res = {'words_ids': words_ids,
            'tmasks': tmasks,

            'vfeats': vfeats,
            'vmasks': vmasks,

            # labels
            'label1ds': label1ds,
            'label2ds': label2ds,

            "concept_inputs": concept_inputs,

            # evaluate
            'se_times': se_times,
            'se_fracs': se_fracs,
            
            
            }

    return res, records



class CCALoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d):
        ious2d = self.scale(ious2d).clamp(0, 1) 
        return F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d)
        )
        
def build_ccaloss(cfg, mask2d):
    min_iou = cfg.MODEL.CCA.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.CCA.LOSS.MAX_IOU
    return CCALoss(min_iou, max_iou, mask2d) 


from utils.utils import generate_2dmask
def train_engine_CCA(model, data, configs):
    from models.loss import lossfun_loc, lossfun_loc2d
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['tmasks'], data['vfeats'], data['vmasks'], data['concept_inputs'])

    label1ds =  data['label1ds']
    label2ds =  data['label2ds']
    logit2D_mask = generate_2dmask(configs.MODEL.CCA.NUM_CLIPS).to(configs.device)

    lossfun_ccaloss = build_ccaloss(configs, logit2D_mask)
    loc_2dloss = lossfun_ccaloss(output["scores2d"], label2ds)
    
    loss = loc_2dloss 
    return loss, output


def infer_CCA(output, configs):
    from utils.engine import infer_basic
    scores2d = output["scores2d"].sigmoid_()
    vmask = output["vmask"]

    outer = torch.triu(scores2d, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res
