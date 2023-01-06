import torch.nn.functional as F
import torch
from models.layers import mask_logits
from torch import nn
from utils.utils import calculate_iou, calculate_iou_accuracy
import numpy as np


# def lossfun_match(m_probs, label_embs, m_labels, vmask):
    
#     ## cross_entropy????
#     m_labels = F.one_hot(m_labels)
#     loss_per_sample = -torch.sum(m_labels * m_probs, dim=-1)
#     m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
#     m_loss = m_loss.mean()
    
#     # add punishment
#     ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
#     ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
#     m_loss += ortho_constraint

#     return m_loss

def lossfun_match(m_probs, label_embs, m_labels, vmask):
    # NLLLoss
    # loss_fun = nn.NLLLoss()
    loss_fun = nn.CrossEntropyLoss()
    m_labels = F.one_hot(m_labels).float()
    m_loss = loss_fun(m_probs, m_labels)
    # m_loss = loss_fun(m_probs.transpose(1,2), m_labels)

    loss_per_sample = -torch.sum(m_labels * m_probs, dim=-1)
    m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
    m_loss = m_loss.mean()
    
    # add punishment
    ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
    ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
    m_loss += ortho_constraint
    return m_loss

def lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask):
    # start_logits = mask_logits(start_logits, vmask)
    # end_logits = mask_logits(end_logits, vmask)

    start_losses = F.cross_entropy(start_logits, s_labels)
    end_losses = F.cross_entropy(end_logits, e_labels)
    loss = (start_losses + end_losses).mean()
    return loss





# def infer_my(start_logits, end_logits, vmask):
#     L = start_logits.shape[1]
#     start_logits = mask_logits(start_logits, vmask)
#     end_logits = mask_logits(end_logits, vmask)

#     start_prob = torch.softmax(start_logits, dim=1) ### !!!
#     end_prob = torch.softmax(end_logits, dim=1)

#     outer = torch.matmul(start_prob.unsqueeze(2),end_prob.unsqueeze(1))
#     pad = nn.ReflectionPad2d(padding=(2, 2, 2, 2))
#     outer = pad(outer).unsqueeze(1)
#     kernel = torch.ones(1, 1, 5, 5).cuda()
#     outer = F.conv2d(outer, kernel, padding=0).squeeze()
#     outer = torch.triu(outer, diagonal=0)

#     _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
#     _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
#     start_frac = (start_index/vmask.sum(dim=1)).cpu().numpy()
#     end_frac = (end_index/vmask.sum(dim=1)).cpu().numpy()
#     return start_frac, end_frac


def append_ious(ious, se_gts, se_props):
    # start_fracs, end_fracs = se_props[:, 0], se_props[:, 1]
    for i in range(len(se_gts)):
        gt_se = se_gts[i]
        prop_se = se_props[i]
        iou = calculate_iou(gt_se, prop_se)
        ious.append(iou)
    return ious

# def append_ious(ious, records, props_frac):
#     start_fracs, end_fracs = props_frac[:, 0], props_frac[:, 1]
#     for record, sp, ep in zip(records, start_fracs, end_fracs):
#         sta_gtfrac = record['s_time']/record["duration"]
#         end_gtfrac = record['e_time']/record["duration"]
#         iou = calculate_iou([sp, ep], [sta_gtfrac, end_gtfrac])
#         ious.append(iou)
    
#     return ious


def get_i345_mi(ious):
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    # mi = torch.mean(ious) * 100.0
    mi = np.mean(ious) * 100.0
    return r1i3, r1i5, r1i5, r1i7, mi




# ### CPL
def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss_cpl(configs, tlogist_prop, words_id, words_mask, tlogist_gt=None):
    P = configs.others.cpl_num_props
    B = tlogist_prop.size(0) // P

    words_mask1 = words_mask.unsqueeze(1) \
        .expand(B, P, -1).contiguous().view(B*P, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(B, P, -1).contiguous().view(B*P, -1)

    nll_loss, acc = cal_nll_loss(tlogist_prop, words_id1, words_mask1)
    nll_loss = nll_loss.view(B, P)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    # if not tlogist_gt:
    #     ref_nll_loss, ref_acc = cal_nll_loss(tlogist_gt, words_id, words_mask) 
    #     final_loss = final_loss + ref_nll_loss.mean()
    #     final_loss = final_loss / 2

    return final_loss


def div_loss_cpl(words_logit, gauss_weight, configs):
    P = configs.others.cpl_num_props
    B = words_logit.size(0) // P
    
    gauss_weight = gauss_weight.view(B, P, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(P).unsqueeze(0).cuda() * configs.others.cpl_div_lambda
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    return div_loss.mean() * configs.others.cpl_div_loss_alhpa


def lossfun_loc2d(scores2d, labels2d, mask2d):
    def scale(iou, min_iou, max_iou):
        return (iou - min_iou) / (max_iou - min_iou)

    labels2d = scale(labels2d, 0.5, 1.0).clamp(0, 1)
    loss_loc2d = F.binary_cross_entropy_with_logits(
        scores2d.squeeze().masked_select(mask2d),
        labels2d.masked_select(mask2d)
    )
    return loss_loc2d