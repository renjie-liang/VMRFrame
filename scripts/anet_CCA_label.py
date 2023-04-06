import numpy as np
from utils.utils import load_json, save_pickle
import torch
from tqdm import tqdm

def convert_2D_1D(scores):
    slogit = np.max(scores, axis=1)
    elogit = np.max(scores, axis=0)
    return slogit, elogit


anet_train_path = "data/anet_i3d_gt/train.json"
data_raw = load_json(anet_train_path)
vids = set([i[0] for i in data_raw])
print(len(vids))
output = np.load("results/anet_CCA_train.npy", allow_pickle=True)
save_dict = []
# for i, _ in zip(output, range(10)): # 42.14 26.71
for i in tqdm(output):
    vid = i[0]
    if vid not in vids:
        continue
    slogit, elogit = convert_2D_1D(i[1])
    slogit, elogit = torch.from_numpy(slogit), torch.from_numpy(elogit)
    slogit = torch.nn.functional.normalize(slogit, dim=0)
    elogit = torch.nn.functional.normalize(elogit, dim=0)
    # print(min(slogit), max(slogit))
    se_logits = torch.stack([slogit, elogit])
    save_dict.append([vid, se_logits])

# check the vids
for i, j in zip(save_dict, data_raw):
    assert i[0] == j[0]

save_pickle(save_dict, "./results/anet_CCA_train_logits.pkl")
    
