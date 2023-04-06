import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm
from utils.utils import load_json

anet_train_path = "data/anet_i3d_gt/train.json"
data_train = load_json(anet_train_path)
ban_result = load_pickle("./results/anet_BAN_train_result.pkl")
vlen = 64

vids = set()
for sample in data_train:
    vid = sample[0]
    vids.add(vid)

vids = ban_result["vids"]
score_pred_1d = ban_result["score_pred_1d"]
prop_s_e = ban_result["prop_s_e"]

save_dict = []
slogit = torch.zeros(64)
elogit = torch.zeros(64)
for k in tqdm(range(len(vids))):
    vid_raw = data_train[k][0]
    vid = vids[k]
    assert vid == vid_raw
    
    score = score_pred_1d[k]
    prop = prop_s_e[k]

    for i, j in zip(prop, score):
        s, e = i
        sl = get_gaussian_weight(s, vlen=64, L=64, alpha=0.1)
        el = get_gaussian_weight(e, vlen=64, L=64, alpha=0.1)
        slogit += sl
        elogit += el 
    slogit = torch.nn.functional.normalize(slogit, dim=0)
    elogit = torch.nn.functional.normalize(elogit, dim=0)
    # plt.plot(elogit)
    # plt.plot(slogit)
    # plt.savefig("./images/{}.jpg".format(vid))
    # plt.cla()
    # break
    # save_dict.append([vids, np.stack([slogit.numpy(), elogit.numpy()])])
    save_dict.append([vid, torch.stack([slogit, elogit])])
save_pickle(save_dict, "./results/anet_BAN_train_logits.pkl")