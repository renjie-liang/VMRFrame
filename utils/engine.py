import torch
from models.layers import mask_logits
from models.loss import lossfun_match, lossfun_loc, append_ious, get_i345_mi, rec_loss_cpl, div_loss_cpl
import torch.nn.functional as F
import torch.nn as nn




def train_engine_SeqPAN(model, data, configs):
    device = configs.device
    _, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
    # prepare features
    vfeats, vmask = vfeats.to(device), vmask.to(device) ### move_to_cuda
    word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
    s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)
    
    # # compute logits
    output= model(word_ids, char_ids, vfeats, vmask, tmask)

    start_logits = output["start_logits"]
    end_logits = output["end_logits"]
    match_score = output["match_score"]
    label_embs = output["label_embs"]

    m_loss = lossfun_match(match_score, label_embs, m_labels, vmask)
    loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
    loss =loc_loss + m_loss

    output["vmask"] = vmask
    return loss, output




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



def scale(iou, min_iou, max_iou):
    return (iou - min_iou) / (max_iou - min_iou)


# def train_engine_BAN(model, indata, configs):
#     # _, data = indata
#     data, info = indata
#     data = {key: value.to(configs.device) for key, value in data.items()}
#     # out = model(data['vfeats'], data['words_ids'], data['vlens'], data['tlens'], data['start_end_offset'])
#     out = model(data)

#     # loss bce
#     scores2d, ious2d, mask2d = out['tmap'], data['iou2d'], out['map2d_mask'],
#     ious2d_scaled = scale(ious2d, configs.loss.min_iou, configs.loss.max_iou).clamp(0, 1)
#     loss_bce = F.binary_cross_entropy_with_logits(
#         scores2d.squeeze().masked_select(mask2d),
#         ious2d_scaled.masked_select(mask2d)
#     )

#     # loss refine
#     final_pred = out['final_pred']
#     pred_s_e_round = out['coarse_pred_round']
#     ious_gt = []
#     for i in range(ious2d_scaled.size(0)):
#         start = pred_s_e_round[i][:, 0]
#         end = pred_s_e_round[i][:, 1] - 1
#         final_ious = ious2d_scaled[i][start, end]
#         ious_gt.append(final_ious)
#     ious_gt = torch.stack(ious_gt)

#     loss_refine = F.binary_cross_entropy_with_logits(
#         final_pred.squeeze().flatten(),
#         ious_gt.flatten()
#     )

#     # distribute differe
#     from models.BAN import temporal_difference_loss

#     dist_idxs =  data['s_e_distribution']
#     td = out['td']
#     td_mask = dist_idxs.sum(dim=1)
#     loss_td = temporal_difference_loss(td, td_mask)


#     # offset loss
#     offset_pred, offset_gt = out['offset'], out['offset_gt'] 
#     offset_pred = offset_pred.reshape(-1, 2)
#     offset_gt = offset_gt.reshape(-1, 2)
#     offset_loss_fun = nn.SmoothL1Loss()
#     loss_offset = offset_loss_fun(offset_pred[:, 0], offset_gt[:, 0]) + offset_loss_fun(offset_pred[:, 1], offset_gt[:, 1])


#     # contrast loss
#     from models.BAN import ContrastLoss

#     map2d_contrasts = data['mask2d_contrast']
#     sen_proj, map2d_proj = out['sen_proj'],  out['map2d_proj']
#     mask2d_pos = map2d_contrasts[:, 0, :, :]
#     mask2d_neg = map2d_contrasts[:, 1, :, :]
#     mask2d_pos = torch.logical_and(mask2d, mask2d_pos)
#     mask2d_neg = torch.logical_and(mask2d, mask2d_neg)
#     loss_contrast = ContrastLoss()(sen_proj, map2d_proj, mask2d_pos, mask2d_neg)


#     loss = loss_bce * configs.loss.bce \
#          + loss_refine * configs.loss.refine \
#          + loss_td * configs.loss.td \
#          + loss_offset * configs.loss.offset \
#          + loss_contrast * configs.loss.contrast
#     return loss, out


def train_engine_BAN(model, indata, configs):
    data, _ = indata
    data = {key: value.to(configs.device) for key, value in data.items()}
    out = model(data['vfeats'], data['words_ids'], data['vlens'], data['tlens'], data['start_end_offset'])

    # loss bce
    scores2d, ious2d, mask2d = out['tmap'], data['iou2ds'], out['map2d_mask'],
    ious2d_scaled = scale(ious2d, configs.loss.min_iou, configs.loss.max_iou).clamp(0, 1)
    loss_bce = F.binary_cross_entropy_with_logits(
        scores2d.squeeze().masked_select(mask2d),
        ious2d_scaled.masked_select(mask2d)
    )

    # loss refine
    final_pred = out['final_pred']
    pred_s_e_round = out['coarse_pred_round']
    ious_gt = []
    for i in range(ious2d_scaled.size(0)):
        start = pred_s_e_round[i][:, 0]
        end = pred_s_e_round[i][:, 1] - 1
        final_ious = ious2d_scaled[i][start, end]
        ious_gt.append(final_ious)
    ious_gt = torch.stack(ious_gt)

    loss_refine = F.binary_cross_entropy_with_logits(
        final_pred.squeeze().flatten(),
        ious_gt.flatten()
    )

    # distribute differe
    from models.BAN import temporal_difference_loss

    dist_idxs =  data['dist_idxs']
    td = out['td']
    td_mask = dist_idxs.sum(dim=1)
    loss_td = temporal_difference_loss(td, td_mask)


    # offset loss
    offset_pred, offset_gt = out['offset'], out['offset_gt'] 
    offset_pred = offset_pred.reshape(-1, 2)
    offset_gt = offset_gt.reshape(-1, 2)
    offset_loss_fun = nn.SmoothL1Loss()
    loss_offset = offset_loss_fun(offset_pred[:, 0], offset_gt[:, 0]) + offset_loss_fun(offset_pred[:, 1], offset_gt[:, 1])


    # contrast loss
    from models.BAN import ContrastLoss

    map2d_contrasts = data['map2d_contrasts']
    sen_proj, map2d_proj = out['sen_proj'],  out['map2d_proj']
    mask2d_pos = map2d_contrasts[:, 0, :, :]
    mask2d_neg = map2d_contrasts[:, 1, :, :]
    mask2d_pos = torch.logical_and(mask2d, mask2d_pos)
    mask2d_neg = torch.logical_and(mask2d, mask2d_neg)
    loss_contrast = ContrastLoss()(sen_proj, map2d_proj, mask2d_pos, mask2d_neg)


    loss = loss_bce * configs.loss.bce \
         + loss_refine * configs.loss.refine \
         + loss_td * configs.loss.td \
         + loss_offset * configs.loss.offset \
         + loss_contrast * configs.loss.contrast
    return loss, out



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
    
    start_frac = (start_index/vmask.sum(dim=1)).cpu().numpy()
    end_frac = (end_index/vmask.sum(dim=1)).cpu().numpy()
    return start_frac, end_frac

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
        
def nms(moments, scores, topk=5, thresh=0.5):
    from models.BAN import iou

    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = torch.zeros_like(ranks).bool()
    numel = suppressed.numel()
    count = 0
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
        count += 1
        if count == topk:
            break
    return moments[~suppressed]


def infer_BAN(output, configs): ## don't consider vmask
    num_clips = configs.model.vlen
    nms_thresh=0.7

    score_pred = output['final_pred'].sigmoid()
    prop_s_e = output['coarse_pred_round']
    res = []
    for idx, score1d in enumerate(score_pred):
        candidates = prop_s_e[idx] / num_clips
        moments = nms(candidates, score1d, topk=1, thresh=nms_thresh)
        res.append(moments[0])
    res = torch.stack(res)
    res = res.cpu().numpy()
    return res
        
