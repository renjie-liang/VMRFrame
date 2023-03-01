import torch
from models import *

def get_loader(dataset, video_features, configs, loadertype):
    dataset_fn = eval(configs.model.name + "Dataset")
    collate_fn = eval(configs.model.name + "Collate")()
    if loadertype == "train":
        shuffle = True
    else:
        shuffle = False

    data_set = dataset_fn(dataset=dataset, video_features=video_features, configs=configs, loadertype=loadertype)
    loader = torch.utils.data.DataLoader(dataset=data_set,  batch_size=configs.train.batch_size, 
                                        shuffle=shuffle, collate_fn=collate_fn)
    return loader

