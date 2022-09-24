import os
import argparse
from tqdm import tqdm
import torch
from models.model import SeqPAN
from models.loss import lossfun_match, lossfun_loc, infer, append_ious, get_i345_mi
from utils.data_gen import load_dataset
# from utils.data_loader import TrainLoader, TestLoader, TrainNoSuffleLoader
from utils.data_utils import load_video_features #load_json, save_json, 
# from utils.runner_utils import eval_test_save, get_feed_dict, write_tf_summary, set_tf_config, eval_test
from datetime import datetime
from utils.utils import load_json, set_seed_config, build_optimizer_and_scheduler, plot_labels, AverageMeter, get_logger
from utils.data_loader import get_loader
import torch.nn.functional as F
from torch import nn
import numpy as np

from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    # parser.add_argument('--log_dir', default=None, type=str, help='log file save path')
    # parser.add_argument('--suffix', default='base', type=str, help='')
    parser.add_argument('--suffix', type=str, default='', help='saved checkpoint suffix')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    return parser.parse_args()


args = parse_args()
configs = EasyDict(load_json(args.config))
configs['suffix'] = args.suffix

set_seed_config(args.seed)
dataset = load_dataset(configs)

configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
print(configs.task)
visual_features = load_video_features(configs.dataset.feature_path, configs.max_pos_len)

# train_loader = TrainLoader(dataset=dataset['train_set'], visual_features=visual_features, configs=configs)
# test_loader = TestLoader(datasets=dataset, visual_features=visual_features, configs=configs)
# train_nosuffle_loader = TrainNoSuffleLoader(datasets=dataset['train_set'], visual_features=visual_features, configs=configs)

train_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="train")
test_loader = get_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs, loadertype="test")
# val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], visual_features, configs)
# test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)
# train_nosuffle_loader = get_test_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
configs.train.num_train_steps = len(train_loader) * configs.train.epochs


model_dir = os.path.join(configs.model_dir, "{}_{}".format(configs.task, configs.suffix))
os.makedirs(model_dir, exist_ok=True)
device = ("cuda" if torch.cuda.is_available() else "cpu" )

def build_load_model(configs, word_vector):
    model = SeqPAN(configs, word_vector)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(device)
    if args.checkpoint:
        model_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(model_checkpoint)
    return model

# init logger and meter
logger = get_logger(model_dir, "eval")
logger.info(configs)
logger.info(args)
lossmeter = AverageMeter()

# train and test
if not args.eval:

    # build model
    model = build_load_model(configs, dataset['word_vector'])
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    best_r1i7, global_step, mi_val_best = -1.0, 0, 0
    for epoch in range(configs.train.epochs):
        model.train()
        lossmeter.reset()
        tbar = tqdm(train_loader)
        ious = []

        for data in tbar:
            records, vfeats, vmask, word_ids, char_ids, tmask, s_labels, e_labels, m_labels = data
            # plot_labels(s_labels, e_labels, m_labels, "SeqPAN")
            # plot_labels(s_labels, e_labels, m_labels, "VSL")
            
            # prepare features
            vfeats, vmask = vfeats.to(device), vmask.to(device) ### move_to_cuda
            word_ids, char_ids, tmask = word_ids.to(device), char_ids.to(device), tmask.to(device)
            s_labels, e_labels, m_labels = s_labels.to(device), e_labels.to(device), m_labels.to(device)
            
            # # compute logits
            start_logits, end_logits, m_probs, label_embs= model(word_ids, char_ids, vfeats, vmask, tmask)
            m_loss = lossfun_match(m_probs, label_embs, m_labels, vmask)
            loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
            loss = m_loss + loc_loss
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
            
            start_logits, end_logits, m_probs, label_embs= model(word_ids, char_ids, vfeats, vmask, tmask)
            m_loss = lossfun_match(m_probs, label_embs, m_labels, vmask)
            loc_loss = lossfun_loc(start_logits, end_logits, s_labels, e_labels, vmask)
            loss = m_loss + loc_loss
            lossmeter.update(loss)
            tbar.set_description("TEST  {:2d}|{:2d} LOSS:{:.4f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))

            start_fracs, end_fracs = infer(start_logits, end_logits, vmask)
            ious = append_ious(ious, records, start_fracs, end_fracs)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        logger.info("TEST |\tmIoU: {:.2f}\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tloss:{:.4f}".format(mi, r1i3, r1i5, r1i7, lossmeter.avg))
        logger.info("")
print("Done!")