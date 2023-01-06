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

