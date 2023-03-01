import os
import json
import easydict
from easydict import EasyDict
import numpy as np
from tqdm import tqdm


def get_vfeat_len(feature_dir, max_pos_len):
    vfeat_lens = {}
    for vid in tqdm(os.listdir(feature_dir), desc="get video feature lengths"):
        tmp = os.path.join(feature_dir, vid)
        ll = np.load(tmp).shape[0]
        vfeat_lens[vid[:-4]] = min(max_pos_len, ll)
    return vfeat_lens 


raw_path =  "data/charades_clean/train.json"
with open(raw_path, 'r') as fr:
    raw_json = json.load(fr)
vfeat_lens = get_vfeat_len(feature_dir="/storage_fast/rjliang/charades/c3d_1024", max_pos_len=64)
 


def time_to_index_my(s, e,  feature_len, duration):
    feature_len = feature_len - 1
    s = round(s / duration * feature_len)
    e = round(e / duration * feature_len)
    return s, e


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


for idx, record in enumerate(raw_json):
    vid = record[0]
    duration = record[1]
    s, e = record[2]
    flen = vfeat_lens[vid]
    s_ind1, e_ind1, _ = time_to_index(s, e, flen, duration)
    s_ind2, e_ind2 = time_to_index_my(s, e, flen, duration)
    
    assert e_ind1 <= flen
    assert e_ind2 <= flen
    assert s_ind1 == s_ind2, "{} {}".format(s_ind1, s_ind2)
    assert e_ind1 == e_ind2, "{} {} {}".format(e_ind1, e_ind2, idx)
    




for i in range(100):

    s = 0.0
    e = 0.0 + i*0.1
    feature_nums = 11
    duration = 10

    s_ind1, e_ind1, _ = time_to_index(s, e, feature_nums, duration)
    s_ind2, e_ind2 = time_to_index_my(s, e, feature_nums, duration)

    print("{:.2f} {:.2f}".format(s, e), end="")
    print("|", s_ind1, e_ind1, "|", s_ind2, e_ind2)


# assert s_ind1 == s_ind2, "{} {}".format(s_ind1, s_ind2)
# assert e_ind1 == e_ind2, "{} {} {}".format(e_ind1, e_ind2, idx)

