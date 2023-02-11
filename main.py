from distutils.command.config import config
import os
import argparse
import torch
from torch import nn

import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from models.loss import append_ious, get_i345_mi
from utils.data_gen import load_dataset
from utils.data_utils import VideoFeatureDict
from utils.utils import load_json, load_yaml, set_seed_config, build_optimizer_and_scheduler, plot_labels, AverageMeter, get_logger, save_best_model
from utils.data_loader import get_loader
from models import *
import yaml

torch.set_printoptions(precision=4, sci_mode=False)
def build_load_model(configs, args, word_vector):
    model = eval(configs.model.name)(configs, word_vector)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(configs.device)
    if args.checkpoint:
        model_checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(model_checkpoint)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--debug', action='store_true', help='only debug')
    parser.add_argument('--suffix', type=str, default='', help='task suffix')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    return parser.parse_args()

args = parse_args()
configs = EasyDict(load_yaml(args.config))
configs['suffix'] = args.suffix

set_seed_config(args.seed)
dataset = load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
visual_features = VideoFeatureDict(configs.paths.feature_path, configs.model.vlen, args.debug, configs.model.sample_type)
train_loader = get_loader(dataset['train_set'], visual_features, configs, loadertype="train")
test_loader = get_loader(dataset['test_set'], visual_features, configs, loadertype="test")
# train_nosuffle_loader = get_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs, loadertype="test")
configs.train.num_train_steps = len(train_loader) * configs.train.epochs

ckpt_dir = os.path.join(configs.paths.ckpt_dir, "{}_{}".format(configs.task, configs.suffix))
os.makedirs(ckpt_dir, exist_ok=True)
device = ("cuda" if torch.cuda.is_available() else "cpu" )
configs.device = device

# init logger and meter
logger = get_logger(ckpt_dir, "eval")
logger.info(args)
logger.info(configs)
lossmeter = AverageMeter()

# glove_emb_path = '/storage/rjliang/1_WeakVMR/BAN-APR/Charades_STA/data/caption/charasdes_sta_captions_glove_embeds.npy'
# glove_emb = np.load(open(glove_emb_path, 'rb'))
# train and test
if not args.eval:
    # build model
    # model = build_load_model(configs, args, glove_emb) 
    model = build_load_model(configs, args, dataset['word_vector'])
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    best_r1i7, global_step, mi_val_best = -1.0, 0, 0
    for epoch in range(configs.train.epochs):
        model.train()
        lossmeter.reset()
        tbar, ious = tqdm(train_loader), []
        for data in tbar:
            inputbatch, records = data
            train_engine = eval("train_engine_" + configs.model.name)
            loss, output = train_engine(model, inputbatch, configs)

            lossmeter.update(loss.item())
            tbar.set_description("TRAIN {:2d}|{:2d} LOSS:{:.6f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.train.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()

            infer_fun = eval("infer_" + configs.model.name)
            props_frac = infer_fun(output, configs)
            ious = append_ious(ious,  inputbatch["se_fracs"], props_frac)
            # ious = append_ious(ious, records, props_frac)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        logger.info("TRAIN|\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(r1i3, r1i5, r1i7, mi, lossmeter.avg))

        model.eval()
        lossmeter.reset()
        tbar = tqdm(test_loader)
        ious, ious_my = [], []

        for data in tbar:
            inputbatch, records = data
            train_engine = eval("train_engine_" + configs.model.name)
            loss, output = train_engine(model, inputbatch, configs)
            lossmeter.update(loss.item())
            tbar.set_description("TEST  {:2d}|{:2d} LOSS:{:.6f}".format(epoch + 1, configs.train.epochs, lossmeter.avg))
            infer_fun = eval("infer_" + configs.model.name)
            props_frac = infer_fun(output, configs)
            ious = append_ious(ious, inputbatch["se_fracs"], props_frac)
            # ious = append_ious(ious, records, props_frac)
        r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
        save_name = os.path.join(ckpt_dir, "best_{}.pkl".format(configs.model.name))
        save_best_model(mi, model, save_name)
        logger.info("TEST |\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(r1i3, r1i5, r1i7, mi, lossmeter.avg))
        logger.info("")

if args.eval:
    # model = build_load_model(configs, args, glove_emb)
    model = build_load_model(configs, args, dataset['word_vector'])
    model.eval()
    lossmeter.reset()
    tbar, ious = tqdm(test_loader), []
    for data in tbar:
        records, _ = data
        train_engine = eval("train_engine_" + configs.model.name)
        loss, output = train_engine(model, records, configs)
        lossmeter.update(loss.item())
        infer_fun = eval("infer_" + configs.model.name)
        props_frac = infer_fun(output, configs)
        ious = append_ious(ious, records["se_fracs"], props_frac)
    r1i3, r1i5, r1i5, r1i7, mi = get_i345_mi(ious)
    logger.info("TEST |\tR1I3: {:.2f}\tR1I5: {:.2f}\tR1I7: {:.2f}\tmIoU: {:.2f}\tloss:{:.4f}".format(r1i3, r1i5, r1i7, mi, lossmeter.avg))
    logger.info("")

print("Done!")