from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
opt = TrainOptions().parse()

torch.set_num_threads(6)


opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id = -1
opt.verbose = False
opt.phase = 'test'


realdir = '/nas_data/zhangzy/uhd_removal_dataset/real/'


eval_dataset_uhd = datasets.UhdTestDataset(opt)

eval_dataloader_uhd = datasets.DataLoader(
    eval_dataset_uhd, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)

"""Main Loop"""
result_dir = './results'
# test on our collected unaligned data or internet images
engine.test(eval_dataloader_uhd, savedir=join(result_dir, opt.name+'_test'))
