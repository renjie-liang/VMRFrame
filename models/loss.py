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
    m_loss = loss_fun(m_probs.transpose(1,2), m_labels)

    m_labels = F.one_hot(m_labels)
    loss_per_sample = -torch.sum(m_labels * m_probs, dim=-1)
    m_loss =torch.sum(loss_per_sample * vmask, dim=-1) / (torch.sum(vmask, dim=-1) + 1e-12)
    m_loss = m_loss.mean()
    
    # add punishment
    ortho_constraint = torch.matmul(label_embs.T, label_embs) * (1.0 - torch.eye(4, device=label_embs.device, dtype=torch.float32))
    ortho_constraint = torch.norm(ortho_constraint, p=2)  # compute l2 norm as loss
    m_loss += ortho_constraint
    return m_loss

def lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask):
    start_logits = mask_logits(start_logits, vmask)
    end_logits = mask_logits(end_logits, vmask)

    start_losses = F.cross_entropy(start_logits, s_labels)
    end_losses = F.cross_entropy(end_logits, e_labels)
    loss = (start_losses + end_losses).mean()
    return loss



def infer(start_logits, end_logits, vmask):
    L = start_logits.shape[1]
    start_logits = mask_logits(start_logits, vmask)
    end_logits = mask_logits(end_logits, vmask)

    start_prob = torch.softmax(start_logits, dim=1)
    end_prob = torch.softmax(end_logits, dim=1)
    
    outer = torch.matmul(start_prob.unsqueeze(2),end_prob.unsqueeze(1))
    outer = torch.triu(outer, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    start_frac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    end_frac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    return start_frac, end_frac


def append_ious(ious, records, start_fracs, end_fracs):
    for record, sp, ep in zip(records, start_fracs, end_fracs):
        sta_gtfrac = record['s_time']/record["duration"]
        end_gtfrac = record['e_time']/record["duration"]
        iou = calculate_iou([sp, ep], [sta_gtfrac, end_gtfrac])
        ious.append(iou)
    
    return ious


def get_i345_mi(ious):
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    # mi = torch.mean(ious) * 100.0
    mi = np.mean(ious) * 100.0
    return r1i3, r1i5, r1i5, r1i7, mi