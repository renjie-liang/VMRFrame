from distutils.command.config import config
import os
import argparse
import torch
from torch import nn

import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from models.model import VSLNet
from models.loss import lossfun_match, lossfun_loc, infer, append_ious, get_i345_mi
from utils.data_gen import load_dataset
from utils.data_utils import load_video_features
from utils.utils import load_json, set_seed_config, build_optimizer_and_scheduler, plot_labels, AverageMeter, get_logger
from utils.data_loader import get_loader
torch.set_printoptions(precision=4, sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--suffix', type=str, default='', help='task suffix')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    return parser.parse_args()


args = parse_args()
configs = EasyDict(load_json(args.config))
configs['suffix'] = args.suffix

set_seed_config(args.seed)
dataset = load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
visual_features = load_video_features(configs.dataset.feature_path, configs.max_pos_len)
train_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="train")
test_loader = get_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs, loadertype="test")
# train_nosuffle_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="test")
configs.train.num_train_steps = len(train_loader) * configs.train.epochs


model_dir = os.path.join(configs.model_dir, "{}_{}".format(configs.task, configs.suffix))
os.makedirs(model_dir, exist_ok=True)
device = ("cuda" if torch.cuda.is_available() else "cpu" )
configs.device = device

def build_load_model(configs, word_vector):
    model = VSLNet(configs, word_vector)
    if args.checkpoint:
        print(args.checkpoint)
        model_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(model_checkpoint)
    model  = model.to(device)       
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    return model

# init logger and meter
logger = get_logger(model_dir, "eval")
logger.info(args)
logger.info(configs)
lossmeter = AverageMeter()

model = build_load_model(configs, dataset['word_vector'])

# train and test
if not args.eval:
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    best_r1i7, global_step, mi_val_best = -1.0, 0, 0
    for epoch in range(configs.train.epochs):
        model.train()
        lossmeter.reset()
        tbar = tqdm(train_loader)
        ious = []
        for data in tbar:
            records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, _ = data
            
            # prepare features
            vfeats, vmask = vfeats.to(device), vmask.to(device) 
            word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
            s_labels, e_labels = s_labels.to(device), e_labels.to(device)
            
            # # compute logits
            start_logits, end_logits = model(word_ids, char_ids, vfeats, vmask, tmask)
            loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
            loss =loc_loss
            lossmeter.update(loss)
            tbar.set_description("TRAIN {:2d}|{:2d} LOSS:{:.4f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.train.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()

            # evaluate
            start_fracs, end_fracs = infer(start_logits, end_logits, vmask)
            ious = append_ious(ious, records, start_fracs, end_fracs)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        logger.info("TRAIN|\tmIoU: {:.2f}\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))

    
        model.eval()
        lossmeter.reset()
        tbar = tqdm(test_loader)
        ious = []

        for data in tbar:
            records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
            vfeats, vmask = vfeats.to(device), vmask.to(device) 
            word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
            s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)
            
            start_logits, end_logits= model(word_ids, char_ids, vfeats, vmask, tmask)
            # m_loss = lossfun_match(m_probs, label_embs, m_labels, vmask)
            loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
            loss = loc_loss
            lossmeter.update(loss)
            tbar.set_description("TEST  {:2d}|{:2d} LOSS:{:.4f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))

            start_fracs, end_fracs = infer(start_logits, end_logits, vmask)
            ious = append_ious(ious, records, start_fracs, end_fracs)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        logger.info("TEST |\tmIoU: {:.2f}\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))
        logger.info("")

else:

    model.eval()
    lossmeter.reset()
    tbar = tqdm(test_loader)
    ious = []
    for data in tbar:
        records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
        vfeats, vmask = vfeats.to(device), vmask.to(device) 
        word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
        s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)
        
        start_logits, end_logits= model(word_ids, char_ids, vfeats, vmask, tmask)
        # m_loss = lossfun_match(m_probs, label_embs, m_labels, vmask)
        loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
        loss = loc_loss
        lossmeter.update(loss)
        tbar.set_description("TEST  {:2d}|{:2d} LOSS:{:.4f}".format(0, 0, lossmeter.avg))

        start_fracs, end_fracs = infer(start_logits, end_logits, vmask)
        ious = append_ious(ious, records, start_fracs, end_fracs)
    r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
    logger.info("TEST |\tmIoU: {:.2f}\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))
    logger.info("")
print("Done!")