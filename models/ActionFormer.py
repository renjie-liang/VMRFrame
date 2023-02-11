import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.ActionFormerlib.meta_archs import PtTransformer

class ActionFormer(nn.Module):
    def __init__(self, configs, word_vec):
        super(ActionFormer, self).__init__()
        actionformer_config = configs.actionformer
        self.net = PtTransformer(**actionformer_config)

    def forward(self, input):
        res = self.net(input)
        return res

def collate_fn_ActionFormer(datas):
    records = {}
    tmp = []
    for d in datas:
        tmp.append(torch.from_numpy(d["se_time"]).squeeze())
    records["se_fracs"] = tmp
    return datas, records


def train_engine_ActionFormer(model, data, configs):
    for i in data:
        i["feats"] =  i["feats"].to(configs.device)
        i["segments"] =  i["segments"].to(configs.device)
        i["labels"] =  i["labels"].to(configs.device)

    loss, output = model(data)
    loss = loss["reg_loss"]
    return loss, output


def infer_ActionFormer(output, configs):
    res = []
    for i in output:
        res.append(i["segments"][0])
    res = torch.stack(res)
    return res