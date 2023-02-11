import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm
ban_result = load_pickle("./results/charades_BAN_train_result.pkl")
vlen = 64

vids = ban_result["vids"]
score_pred_1d = ban_result["score_pred_1d"]
prop_s_e = ban_result["prop_s_e"]


# topk = 5
# for i in range(len(vids)):
#     vid = vids[i]
#     score = score_pred_1d[i]
#     prop = prop_s_e[i]
#     # score_prop = []
#     # for i, j in zip(score, prop):
#     #     score_prop.append([i, j])
#     # score_prop = sorted(score_prop, key=lambda x: x[0], reverse=True)

#     prop = torch.as_tensor(prop)
#     score = torch.as_tensor(score)
#     moments, score_rank = nms(prop, score, topk=5, thresh=0.9)
#     moments_rank = moments[:topk].numpy()
#     score_rank = score_rank[:topk].numpy()
    
save_dict = []
slogit = torch.zeros(64)
elogit = torch.zeros(64)
for k in tqdm(range(len(vids))):
    vid = vids[k]
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
    print(slogit.shape)
    # plt.plot(elogit)
    # plt.plot(slogit)
    # plt.savefig("./images/{}.jpg".format(vid))
    # plt.cla()
    # break
    # save_dict.append([vids, np.stack([slogit.numpy(), elogit.numpy()])])
    save_dict.append([vid, torch.stack([slogit, elogit])])
save_pickle(save_dict, "./results/charades_BAN_train_logits.pkl")