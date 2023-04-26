import numpy as np
from utils.utils import load_pickle, save_pickle
from utils.utils import get_gaussian_weight
from matplotlib import pyplot as plt
import torch
import pickle
from tqdm import tqdm

in_path, save_path = "./results/tacos_SeqPAN_train_result.pkl", "./results/tacos_SeqPAN_train_logits.pkl"
# in_path, save_path, vlen = "./results/charades_SeqPAN_train_result.pkl",  "./results/charades_SeqPAN_train_logits2.pkl", 48
# in_path, save_path, vlen = "./results/anet_SeqPAN_train_result.pkl", "./results/anet_SeqPAN_train_logits2.pkl", 64

seqpan_result =  load_pickle(in_path)
save_dict = []
for sample in tqdm(seqpan_result):
    vid, vlen, se_logits = sample["vid"], sample["vlen"], sample["prop_logits"]
    se_logits = torch.from_numpy(np.stack(se_logits))
    se_logits = se_logits[:int(vlen)]
    se_logits = torch.sigmoid(se_logits)
    save_dict.append([vid, se_logits])

    # plt.plot(se_logits[1, :])
    # plt.plot(se_logits[0, :])
    # plt.savefig("./images/{}.jpg".format(vid))
    # plt.cla()
    # break
save_pickle(save_dict, save_path)
print(in_path, save_path)