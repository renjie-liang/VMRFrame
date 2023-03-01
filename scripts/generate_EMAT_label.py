import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm
emat_result = load_pickle("./results/charades_EMAT_train.pkl")

save_dict = []
for sample in tqdm(emat_result):
    vid, se_logits, vlen = sample
    se_logits = se_logits[:int(vlen)]
    se_logits = torch.sigmoid(se_logits)
    save_dict.append([vid, se_logits])

    # plt.plot(se_logits[1, :])
    # plt.plot(se_logits[0, :])
    # plt.savefig("./images/{}.jpg".format(vid))
    # plt.cla()
    # break
save_pickle(save_dict, "./results/charades_EMAT_train_logits.pkl")