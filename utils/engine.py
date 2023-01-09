import torch
from models.layers import mask_logits
from models.loss import lossfun_match, lossfun_loc, append_ious, get_i345_mi, rec_loss_cpl, div_loss_cpl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



def train_engine_CPL(model, data, configs):
    device = configs.device
    _, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
    # prepare features
    vfeats, vmask = vfeats.to(device), vmask.to(device) ### move_to_cuda
    word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
    s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)

    # # compute logits
    output= model(word_ids, char_ids, vfeats, vmask, tmask)


    loss_rec = rec_loss_cpl(configs=configs, tlogist_prop=output["words_logit"], 
                        words_id= output["word_ids"], words_mask=output["words_mask"], tlogist_gt=None)
    loss_div = div_loss_cpl(words_logit=output["words_logit"], gauss_weight= output["gauss_weight"], configs=configs)
    loss = loss_rec + loss_div

    output["vmask"] = vmask
    return loss, output







def infer_basic(start_logits, end_logits, vmask):
    L = start_logits.shape[1]
    start_logits = mask_logits(start_logits, vmask)
    end_logits = mask_logits(end_logits, vmask)

    start_prob = torch.softmax(start_logits, dim=1) ### !!!
    end_prob = torch.softmax(end_logits, dim=1)
    
    outer = torch.matmul(start_prob.unsqueeze(2),end_prob.unsqueeze(1))
    outer = torch.triu(outer, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res



def infer_basic2d(scores2d, logit2D_mask, vmask):
    scores2d = scores2d.sigmoid_() * logit2D_mask

    outer = torch.triu(scores2d, diagonal=0)
    _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
    _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
    
    sfrac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    efrac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    res = np.stack([sfrac, efrac]).T
    return res


def infer_SeqPAN(output, configs):
    start_logits = output["start_logits"]
    end_logits = output["end_logits"]
    vmask = output["vmask"]
    start_frac, end_frac = infer_basic(start_logits, end_logits, vmask)
    return start_frac, end_frac



def infer_CPL(output, configs): ## don't consider vmask
    from models.cpl_lib import cal_nll_loss

    P = configs.others.cpl_num_props
    B = output['words_logit'].size(0)  // P

    tmask_props = torch.repeat_interleave(output['words_mask'], P, dim=0)
    word_ids = torch.repeat_interleave(output['word_ids'], P, dim=0)
    nll_loss, acc = cal_nll_loss(output['words_logit'], word_ids, tmask_props)
    idx = nll_loss.view(B, P).argsort(dim=-1)

    width = output['width'].view(B, P).gather(index=idx, dim=-1)
    center = output['center'].view(B, P).gather(index=idx, dim=-1)
    
    selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                    torch.clamp(center+width/2, max=1)], dim=-1)
    selected_props = selected_props.detach().cpu().numpy()
    start_fracs, end_fracs = selected_props[:, 0, 0], selected_props[:, 0, 1] 

    return start_fracs, end_fracs
        
