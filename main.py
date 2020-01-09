#----------------------------------------
# OrthDNNs for Pytorch
# 
# Copyright 2018 Yuxin Wen
#----------------------------------------
from __future__ import division
import time
begin = time.time()
import argparse
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.optim

from Data_loader import Data_loader
from Models import *
from LossF import *
from Optim import *
import Trainer
import Utility

print('The time of requring needed packages:\t{0}'.format(Utility.format_time(time.time() - begin)))

#----------------------------------------
# parser loader
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset',  default='CIFAR10', metavar = 'NAME', help='|CIFAR10|CIFAR100|ImageNet')
parser.add_argument('--dataset_dir',  default='Dataset/', metavar='DIR', help='path to dataset')
parser.add_argument('--decrese_sample_rate',  default=1, type=int,  help='reamian 1/decrese_sample_rate samples(1 for whole dataset)')
parser.add_argument('--save', metavar='DIR', default='Exps/', help='path to Exps')
#----------------------------------------------------------Training options-----------------------------------------------------------
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--nEpoch', default=300, type=int, metavar='N', help='number of total epochs to train')
#------------------------------------------------------Empirical architectural options------------------------------------------------
parser.add_argument('-nGPU', '--n_GPU_to_use', default=1, type=int, metavar='N', help='multi GUP or not (default: 1)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='PreActResNet', help='model architecture (|PreActResNet|ResNext|ReseNet|)')
parser.add_argument('--resume_dir', type=str, metavar='PATH', help='path to latest checkpoint (default: None)')
#-----------------------------------------------------Optimization algorithms options-------------------------------------------------
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr_decay_method', default='exp', type=str, metavar='STR', help='exp|noDecay')
parser.add_argument('-lr', '--lr_base', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_end', default=0.001, type=float, metavar='LR', help='ending learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-wd','--weight_decay',  default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-stiefel','--stiefel_flag',  dest='stiefel_flag', action='store_true', help='to optim on stiefel or not')
parser.add_argument('-svb','--svb_flag',  dest='svb_flag', action='store_true', help='to use svb or not')
parser.add_argument('-iter_svb','--iter_svb_flag',  dest='iter_svb_flag', action='store_true', help='')
parser.add_argument('--svb_factor', default=0.5, type=float, help='epsilon of svb')
parser.add_argument('--svb_epoch', default=1, type=int, metavar='N', help='perform SVB every N epoch')
parser.add_argument('-bbn','--bbn_flag',  dest='bbn_flag', action='store_true', help='to use bbn or not')
parser.add_argument('--bbn_factor', default=0.2, type=float, help='epsilon of bbn')
parser.add_argument('--bbn_type', default='bbn', type=str, help='rel|abs|bbn')
parser.add_argument('--is_soft_regu', dest='is_soft_regu', action='store_true', help='adding W^T*W as Regu or not')
parser.add_argument('--is_SRIP', dest='is_SRIP', action='store_true', help='')
parser.add_argument('--soft_lambda', default=1e-4, type=float, help='the tradeoff between CrossEntropyLoss and SoftOrthDNNs')
#--------------------------------------------------------------Miscellaneous----------------------------------------------------------
parser.add_argument('-p', '--print_freq', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--is_evaluate', dest='is_evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--random_seed', default=0, type=int, help='')


args = parser.parse_args()
best_prec1 = 0
if not os.path.exists(args.dataset_dir):
    os.mkdir(args.dataset_dir)
if not os.path.exists(args.save):
    os.mkdir(args.save)
seed = args.random_seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)   

#----------------------------------------
# data loader
#----------------------------------------
begin = time.time()

data_loader = Data_loader.Data_loader(args.dataset_dir, args.batch_size, args.workers, valid_ratio=0)
if args.dataset == 'CIFAR10':
    dataloaders = data_loader.get_CIFAR10_loader()
    num_classes = 10
else:
    assert False, 'DO NOT support such dataset.'

print('The time of init data_loader:\t{0}'.format(Utility.format_time(time.time() - begin)))
#----------------------------------------
# init model loader, criterion and optimizer
#----------------------------------------
begin = time.time()
if args.arch == 'PreActResNet20' and args.dataset == 'CIFAR10':
    model = PreActResNet20(args.dataset)
elif args.arch == 'PreActResNet20_WOBN' and args.dataset == 'CIFAR10':
    model = PreActResNet20_WOBN()
elif args.arch == 'ConvNet' and args.dataset == 'CIFAR10':
    model = ConvNet20()
elif args.arch == 'ConvNet_WOBN' and args.dataset == 'CIFAR10':
    model = ConvNet20_WOBN()
elif args.arch == 'PreActResNet68' and args.dataset == 'CIFAR10':
    model = PreActResNet68()
else:
    raise Exception('DO NOT support such arch.')

nElement = 0
for m in model.modules():
    if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
      nElement += sum([p.data.nelement() for p in m.parameters()])
print('********  Number of model parameters: {}  ********'.format(nElement))

model = torch.nn.DataParallel(model,device_ids=range(0, args.n_GPU_to_use)).cuda()

if args.is_soft_regu:
    print('Using |WW-I| loss.')
    criterion = CrossEntropyLossWithOrtho(method = 'FNorm').cuda()
elif args.is_SRIP:
    print('Using SRIP loss.')
    criterion = CrossEntropyLossWithOrtho(method = 'SRIP').cuda()
else:
    print('Using pure CE loss.')
    criterion = nn.CrossEntropyLoss().cuda()

if args.stiefel_flag:
    print('Using Stiefel optimization.')
    optimizer = msgd_Stiefel(model.parameters(), args.lr_base,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
else:
    print('Using SGD optimization.')
    optimizer = torch.optim.SGD(model.parameters(), args.lr_base,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

start_epoch = args.start_epoch
if args.resume_dir:
    training_aux = Utility.Training_aux(args.resume_dir)
    start_epoch, best_prec1 = training_aux.load_checkpoint(model, optimizer, is_best = True)

print('The time of init model, criterion and optimizer:{0}\t'.format(Utility.format_time(time.time() - begin)))
#----------------------------------------
# creating trainer
#----------------------------------------
begin = time.time()
trainer = Trainer.Trainer(train_loader = dataloaders['train'], val_loader = dataloaders['val'],
                        model = model, criterion = criterion, optimizer = optimizer, 
                        nEpoch = args.nEpoch, lr_base = args.lr_base, lr_end=args.lr_end, lr_decay_method = args.lr_decay_method, 
                        is_soft_regu = args.is_soft_regu, is_SRIP = args.is_SRIP, soft_lambda = args.soft_lambda, 
                        svb_flag = args.svb_flag, iter_svb_flag = args.iter_svb_flag, svb_factor = args.svb_factor, 
                        bbn_flag = args.bbn_flag, bbn_factor = args.bbn_factor, bbn_type = args.bbn_type,
                        fsave = args.save, print_freq = args.print_freq)

print('The time of init trainer:\t{0}'.format(Utility.format_time(time.time() - begin)))
#----------------------------------------
# mian
#---------------------------------------- 
def main():
    global args, start_epoch, best_prec1
    if args.is_evaluate == True:
        begin = time.time()
        print('Evaluating on testing set:')
        trainer.validate(0)
        print('==> The time of evaluating:\t{0}'.format(Utility.format_time(time.time() - begin)))
        return None

    begin = time.time()
    print('=> Strat training the model')
    # for pretraining (standard model)
    for epoch in range(start_epoch, args.nEpoch):
        trainer.adjust_learning_rate(epoch, scheduler=None)
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save_checkpoint(epoch)
        svb_start_time = time.time()
        if (trainer.svb_flag) and epoch != (trainer.nEpoch -1):
            print('=>performing epoch SVB and BBN')
            trainer.fcConvWeightReguViaSVB()
            print('==> The time of the svb process: \t{0}'.format(Utility.format_time(time.time() - svb_start_time)))

    best_prec1 = trainer.best_prec1 or best_prec1
    print('train Best accuracy: ', best_prec1)
    print('==> The time of the pretrain process:\t{0}'.format(Utility.format_time(time.time() - begin)))

#--------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
