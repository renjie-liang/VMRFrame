import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from models.layers import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, Conv1D, SeqPANPredictor
from models.layers import DualAttentionBlock
from models.SeqPAN import SeqPAN
from models.BAN import BAN

class BaseFast_BAN_PreTrain(nn.Module):
    def __init__(self, configs, word_vectors):
        super(BaseFast_BAN_PreTrain, self).__init__()
        self.configs = configs
        dim = configs.model.dim
        droprate = configs.model.droprate
        max_pos_len = self.configs.model.vlen

        ## ---------- student ------
        self.text_encoder = Embedding(num_words=configs.num_words, num_chars=configs.num_chars, out_dim=dim,
                                       word_dim=configs.model.word_dim, char_dim=configs.model.char_dim, 
                                       word_vectors=word_vectors, droprate=droprate)
        self.video_affine = VisualProjection(visual_dim=configs.model.vdim, dim=dim, droprate=droprate)
        self.feat_encoder = FeatureEncoder(dim=dim, kernel_size=7, num_layers=4, max_pos_len=max_pos_len, droprate=droprate)
        self.q2v_attn = CQAttention(dim=dim, droprate=droprate)
        self.v2q_attn = CQAttention(dim=dim, droprate=droprate)
        self.cq_cat = CQConcatenate(dim=dim)
        self.match_conv1d = Conv1D(in_dim=dim, out_dim=4)
        lable_emb = torch.empty(size=[dim, 4], dtype=torch.float32)
        lable_emb = torch.nn.init.orthogonal_(lable_emb.data)
        self.label_embs = nn.Parameter(lable_emb, requires_grad=True)
        self.predictor = SeqPANPredictor(configs)

        # -------- teacher ban ----------
        configs.teacher0.num_words = configs.num_words
        configs.teacher0.num_chars = configs.num_chars
        self.teach_model = BAN(configs.teacher0, word_vectors)
        print(configs.teacher0.model.checkpoint)
        model_checkpoint = torch.load(configs.teacher0.model.checkpoint)
        self.teach_model.load_state_dict(model_checkpoint)
        # Freeze the parameters of the teacher model
        for param in self.teach_model.parameters():
            param.requires_grad = False

    def forward(self, word_ids, char_ids, vfeat_in, vmask, tmask):
        torch.cuda.synchronize()
        start = time.time()
        B = vmask.shape[0]


        # #  ----------------------- student ----------------
        tfeat = self.text_encoder(word_ids, char_ids)
        vfeat = self.video_affine(vfeat_in)
        vfeat = self.feat_encoder(vfeat)
        tfeat = self.feat_encoder(tfeat)

        t2v_feat = self.q2v_attn(vfeat, tfeat, vmask, tmask)
        v2t_feat = self.v2q_attn(tfeat, vfeat, tmask, vmask)
        fuse_feat = self.cq_cat(t2v_feat, v2t_feat, tmask)


        match_logits = self.match_conv1d(fuse_feat)
        match_score = F.gumbel_softmax(match_logits, tau=0.3)
        soft_label_embs = torch.matmul(match_score, torch.tile(self.label_embs, (B, 1, 1)).permute(0, 2, 1))
        fuse_feat = (fuse_feat + soft_label_embs) * vmask.unsqueeze(2)
        slogits, elogits = self.predictor(fuse_feat, vmask)


        torch.cuda.synchronize()
        end = time.time()
        consume_time =  end - start


        res_t0 = self.teach_model(word_ids, char_ids, vfeat_in, vmask, tmask)
        slogits_t0 = res_t0["slogits"]
        elogits_t0 = res_t0["elogits"]


        return {    "slogits_t0": slogits_t0,
                    "elogits_t0": elogits_t0,

                    "slogits": slogits,
                    "elogits": elogits,
                    "match_score" : match_score,
                    "label_embs" : self.label_embs,

                    "vmask" : vmask,
                    "consume_time": consume_time,
                    }


def collate_fn_BaseFast_BAN_PreTrain(datas):
    from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
    from utils.utils import convert_length_to_mask

    records, se_times, se_fracs = [], [], []
    vfeats, words_ids, chars_ids = [], [], []
    label1ds, NER_labels = [], []
    max_vlen = datas[0]["max_vlen"]
    for d in datas:
        records.append(d["record"])
        vfeats.append(d["vfeat"])
        words_ids.append(d["words_id"])
        label1ds.append(d["label1d"])
        se_times.append(d["se_time"])
        se_fracs.append(d["se_frac"])
        chars_ids.append(d["chars_id"])
        NER_labels.append(d['NER_label'])
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
            'NER_labels': NER_labels,

            # evaluate
            'se_times': se_times,
            'se_fracs': se_fracs,
            # 'vname': vname,
            # 'sentence': sentence
            }

    return res, records


import time
from models.loss import lossfun_softloc
def train_engine_BaseFast_BAN_PreTrain(model, data, configs):
    from models.loss import lossfun_loc, lossfun_match
    data = {key: value.to(configs.device) for key, value in data.items()}
    output = model(data['words_ids'], data['char_ids'], data['vfeats'], data['vmasks'], data['tmasks'])

    label1ds =  data['label1ds']
    vmasks =  data['vmasks']

    slogits_t0 = output["slogits_t0"]
    elogits_t0 = output["elogits_t0"]

    slogits = output["slogits"]
    elogits = output["elogits"]
    loc_loss = lossfun_loc(slogits, elogits, label1ds[:, 0, :], label1ds[:, 1, :], vmasks)
    m_loss = lossfun_match(output["match_score"], output["label_embs"],  data["NER_labels"],  vmasks)
    loss = loc_loss + m_loss

    # --- KL loss ----
    loss_student_t0 = lossfun_softloc(slogits, elogits, slogits_t0, elogits_t0, vmasks, configs.loss.temperature)

    loss = loss + loss_student_t0
    return loss, output


def infer_BaseFast_BAN_PreTrain(output, configs):
    from utils.engine import infer_basic
    vmask = output["vmask"]
    res = infer_basic(output["slogits"], output["elogits"], vmask)
    return res