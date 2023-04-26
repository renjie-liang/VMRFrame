import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm
in_path, save_path, vlen = "./results/tacos_BAN_train_result.pkl", "./results/tacos_BAN_train_logits.pkl", 128
# in_path, save_path, vlen = "./results/charades_BAN_train_result.pkl",  "./results/charades_BAN_train_logits2.pkl", 48
# in_path, save_path, vlen = "./results/anet_BAN_train_result.pkl", "./results/anet_BAN_train_logits2.pkl", 64

ban_result =  load_pickle(in_path)
vids = ban_result["vids"]
score_pred_1d = ban_result["score_pred_1d"]
prop_s_e = ban_result["prop_s_e"]
    
save_dict = []
slogit = torch.zeros(vlen)
elogit = torch.zeros(vlen)
for k in tqdm(range(len(vids))):
    vid, score, prop = vids[k], score_pred_1d[k], prop_s_e[k]
    if "tacos" in in_path:
        vid = vid[:-4]
    for i, j in zip(prop, score):
        s, e = i
        sl = get_gaussian_weight(s, vlen=vlen, L=vlen, alpha=0.1)
        el = get_gaussian_weight(e, vlen=vlen, L=vlen, alpha=0.1)
        slogit += sl * j
        elogit += el * j
    slogit = torch.nn.functional.normalize(slogit, dim=0)
    elogit = torch.nn.functional.normalize(elogit, dim=0)
    # plt.plot(elogit)
    # plt.plot(slogit)
    # plt.savefig("./images/{}.jpg".format(vid))
    # print(vid)
    # plt.cla()
    # break
    # save_dict.append([vids, np.stack([slogit.numpy(), elogit.numpy()])])
    save_dict.append([vid, torch.stack([slogit, elogit])])
save_pickle(save_dict, save_path)
print(in_path, save_path)
