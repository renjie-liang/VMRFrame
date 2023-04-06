import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm
seqpan_result = load_pickle("./results/charades_GMD_result.pkl")

min_, max_=  100, 0
save_dict = []
for sample in tqdm(seqpan_result):
    vid, vlen, se_logits = sample["vid"], sample["vlen"], sample["prop_logits"]
    se_logits = torch.from_numpy(np.stack(se_logits))
    se_logits = se_logits[:int(vlen)]
    min_ = min(se_logits.min(), min_)
    max_ = max(se_logits.max(), max_)
    save_dict.append([vid, se_logits])

    # plt.plot(se_logits[1, :])
    # plt.plot(se_logits[0, :])
    # img_path = "tmp0.jpg"
    # # plt.savefig("./images/{}.jpg".format(vid))
    # plt.savefig(img_path); plt.cla()
    # break
save_pickle(save_dict, "./results/charades_GMD_train_label.pkl")
print(min_, max_)