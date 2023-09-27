from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.set_num_threads(5)

opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 0
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to 


datadir_syn = opt.dataroot


train_dataset = datasets.UhdDataset(opt)

train_datasetloader = datasets.DataLoader(
    train_dataset, batch_size=1, shuffle=False, 
    num_workers=opt.nThreads, pin_memory=True)


# 7643 :  : 90 = 7 3     3000 90    
# realdir = '/nas_data/zhangzy/uhd_removal_dataset/real/'
# datadir_real = realdir
# train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
# train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [1, 0])

# train_dataloader_fusion = datasets.DataLoader(
#     train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
#     num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""
engine = Engine(opt)

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

# if opt.resume:
    # res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

# define training strategy 
engine.model.opt.lambda_gan = 0
# engine.model.opt.lambda_gan = 0.01
set_learning_rate(0.0002)
while engine.epoch < 100:
    if engine.epoch == 20:
        engine.model.opt.lambda_gan = 0.01 # gan loss is added after epoch 20
    # if engine.epoch == 30:
        # set_learning_rate(5e-5)
    # if engine.epoch == 40:
        # set_learning_rate(1e-5)
    # if engine.epoch == 50:
    #     ratio = [0.5, 0.5]
    #     print('[i] adjust fusion ratio to {}'.format(ratio))
    #     train_dataset_fusion.fusion_ratios = ratio
        # set_learning_rate(5e-5)
    # if engine.epoch == 50:
        # set_learning_rate(1e-5)

    # engine.train(train_dataloader_fusion)
    engine.train(train_datasetloader)
    
    # if engine.epoch % 5 == 0:
        # engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        # engine.eval(eval_dataloader_real, dataset_name='testdata_real20')
    # if engine.epoch % 2 == 0:
    # engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
    # engine.eval(eval_dataloader_real, dataset_name='testdata_real20')
