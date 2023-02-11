import numpy as np
from utils.utils import load_json
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from utils.utils import idx_time, calculate_iou
def norm(x):
    return x / np.linalg.norm(x)



result_test = np.load("results/anet_CCA_test.npy", allow_pickle=True)
result_train = np.load("results/anet_CCA_train.npy", allow_pickle=True)
anet_train_path = "data/anet_c3d_gt/train.json"
anet_test_path = "data/anet_c3d_gt/test.json"
data_train = load_json(anet_train_path)
data_test = load_json(anet_test_path)
print(len(result_test), len(data_test))
print(len(result_train), len(data_train))


def convert_2D_1D(scores):
    slogit = np.max(scores, axis=1)
    elogit = np.max(scores, axis=0)
    slogit = norm(slogit)
    elogit = norm(elogit)
    return slogit, elogit


save_list = []

miou = []
# for r, i in zip(result_test, data_test): # 44.62 27.76
for r, i in zip(result_train, data_train): # 42.14 26.71
    assert r[0] == i[0]
    vid, duration, segt, _ = i
    slogit, elogit = convert_2D_1D(r[1])
    sidx, eidx = np.argmax(slogit),  np.argmax(elogit)
    stime, etime = idx_time([sidx, eidx], duration, 64)
    tmp = calculate_iou([stime, etime], segt)
    save_list.append([vid, np.stack([slogit, elogit])])
    miou.append(tmp)


miou = np.array(miou)
r1_07 = sum(miou>0.7) / len(miou)
miou = np.mean(miou)
print(miou, r1_07) 

np.save("./anet_CCA_train_logits", save_list, allow_pickle=True)