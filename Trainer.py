from __future__ import division
import time
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from Utility import Average_meter
from Utility import Training_aux
#from Utility import progress_bar

class Trainer(object):
    """a method that packaging dataloader and model and optim_methods"""
    """the model are trained here"""
    """the mixup operation and data_agu operation are perform here"""
    def __init__(self, train_loader, val_loader, model, criterion, 
            optimizer, nEpoch, lr_base = 0.1, lr_end = 0.001, lr_decay_method = 'exp',
            is_soft_regu=False, is_SRIP=False, soft_lambda = 1e-4, 
            svb_flag = False, iter_svb_flag=False, svb_factor = 0.5, 
            bbn_flag = False, bbn_factor = 0.2, bbn_type = 'rel', 
            fsave = './Save', print_freq = 10, is_evaluate = False, dataset = 'CIFAR10'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.nEpoch = nEpoch
        self.lr_base = lr_base
        self.lr_end = lr_end
        self.lr_decay_method = lr_decay_method

        self.is_soft_regu = is_soft_regu
        self.is_SRIP = is_SRIP
        self.soft_lambda = soft_lambda

        self.svb_flag = svb_flag
        self.iter_svb_flag = iter_svb_flag
        self.svb_factor = svb_factor
        self.bbn_flag = bbn_flag
        self.bbn_factor = bbn_factor
        self.bbn_type = bbn_type

        self.training_aux = Training_aux(fsave)
        self.is_evaluate = is_evaluate
        self.print_freq = print_freq

        self.best_prec1 = 0

    def train(self, epoch):
        """Train for one epoch on the training set"""
        batch_time = Average_meter()
        data_time = Average_meter()
        losses = Average_meter()
        top1 = Average_meter()
        top5 = Average_meter()

        # switch to train mode
        self.model.train()
        begin = time.time()

        for i, (image, target) in enumerate(self.train_loader):
            batch_size= image.size(0)
            # measure data loading time
            data_time.update(time.time() - begin)

            image = image.cuda()
            input_var = Variable(image)
            target = target.cuda()
            target_var = Variable(target)
            
            output = self.model(input_var)

            if self.is_soft_regu or self.is_SRIP:
                loss = self.criterion(output, target_var, self.model, self.soft_lambda)
            else:
                loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.training_aux.accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - begin)

            if i % self.print_freq == 0:
                #progress_bar(i, len(self.train_loader), 'Loss: {loss.avg:.4f} | Prec@1 {top1.avg:.3f} | Prec@5 {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.avg:.3f}\t'
                    'Data {data_time.avg:.3f}\t'
                    'Loss {loss.avg:.4f}\t'
                    'Prec@1 {top1.avg:.3f}\t'
                    'Prec@5 {top5.avg:.3f}'.format(
                     epoch, i, len(self.train_loader), batch_time=batch_time,
                     data_time=data_time, loss=losses, top1=top1, top5=top5))

            begin = time.time()
            if (self.iter_svb_flag) and epoch != (self.nEpoch -1) and i != (self.train_loader.__len__() -1):
                self.fcConvWeightReguViaSVB()

        self.training_aux.write_err_to_file(epoch = epoch, top1 = top1, top5 = top5, trn_loss = losses, mode = 'train')
        
        return 

    def validate(self, epoch, img_size=320):
        """Perform validation on the validation set"""
        batch_time = Average_meter()
        losses = Average_meter()
        top1 = Average_meter()
        top5 = Average_meter()

        self.model.eval()
        begin = time.time()

        with torch.no_grad():
            for i, (raw_img, raw_label) in enumerate(self.val_loader):
                raw_label = raw_label.cuda()
                raw_img = raw_img.cuda()
                input_var = Variable(raw_img)
                target_var = Variable(raw_label)
                # compute output

                output = self.model(input_var)
                # measure accuracy and record loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = self.training_aux.accuracy(output.data, raw_label, topk=(1, 5))
                top1.update(prec1.item(), raw_img.size(0))
                top5.update(prec5.item(), raw_img.size(0))
                losses.update(loss.data.item(), raw_img.size(0))
                # measure elapsed time
                batch_time.update(time.time() - begin)

                if i % self.print_freq == 0:
                    #progress_bar(i, len(self.train_loader), 'Loss: {loss.avg:.4f} | Prec@1 {top1.avg:.3f} | Prec@5 {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.avg:.3f}\t'
                        'Loss {loss.avg:.4f}\t'
                        '{top1.avg:.3f}\t'
                        '{top5.avg:.3f}'.format(
                        i, len(self.val_loader), batch_time=batch_time, 
                        loss=losses, top1=top1, top5=top5))
                begin = time.time()
            print(' * Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                    .format(loss=losses, top1=top1, top5=top5))

            self.is_best = top1.avg > self.best_prec1
            self.best_prec1 = max(top1.avg, self.best_prec1)

            if self.is_evaluate:
                return top1.avg
            else:
                self.training_aux.write_err_to_file(epoch = epoch, top1 = top1, top5 = top5, mode = 'val')
                return top1.avg

    def adjust_learning_rate(self, epoch, warm_up_epoch = 0,scheduler=None):
        """Sets the learning rate to the initial LR decayed by 10 after 0.5 and 0.75 epochs"""
        if self.lr_decay_method == 'exp':
            lr = self.lr_base
            if epoch < warm_up_epoch:
                lr = 0.001 + (self.lr_base - 0.001) * epoch / warm_up_epoch
            if epoch >= warm_up_epoch:
                lr_series = torch.logspace(math.log(self.lr_base, 10), math.log(self.lr_end, 10), int(self.nEpoch/2))
                lr = lr_series[int(math.floor((epoch-warm_up_epoch)/2))]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.lr_decay_method == 'noDecay':
            lr = self.lr_base
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        print('lr:{0}'.format(self.optimizer.param_groups[-1]['lr']))
        return 

    def save_checkpoint(self, epoch, save_flag = 'learning', filename = False):
        if save_flag == 'standard':
            model = self.standard_model
            optimizer = self.standard_optimizer
        elif save_flag == 'learning':
            model = self.model
            optimizer = self.optimizer
        else:
            raise Exception('save_flag should be one of standard or learning')

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': self.best_prec1,
            'optimizer' : optimizer.state_dict(),
            }
        fname = filename or 'checkpoint' + '.pth.tar'

        self.training_aux.save_checkpoint(state = state, is_best = self.is_best, filename=fname)
        return 

    def fcConvWeightReguViaSVB(self):
        for m in self.model.modules():
            #svb
            if self.svb_flag == True:
                if isinstance(m,nn.Conv2d):
                    tmpbatchM = m.weight.data.view(m.weight.data.size(0), -1).t().clone()
                    try:
                        tmpU, tmpS, tmpV = torch.svd(tmpbatchM)
                    except:
                        tmpbatchM = tmpbatchM[np.logical_not(np.isnan(tmpbatchM))]
                        tmpbatchM = tmpbatchM.view(m.weight.data.size(0), -1).t()
                        tmpU, tmpS, tmpV = np.linalg.svd(tmpbatchM.cpu().numpy())
                        tmpU = torch.from_numpy(tmpU).cuda()
                        tmpS = torch.from_numpy(tmpS).cuda()
                        tmpV = torch.from_numpy(tmpV).cuda()

                    for idx in range(0, tmpS.size(0)):
                        if tmpS[idx] > (1+self.svb_factor):
                            tmpS[idx] = 1+self.svb_factor
                        elif tmpS[idx] < 1/(1+self.svb_factor):
                            tmpS[idx] = 1/(1+self.svb_factor)
                    tmpbatchM = torch.mm(torch.mm(tmpU, torch.diag(tmpS.cuda())), tmpV.t()).t().contiguous()
                    m.weight.data.copy_(tmpbatchM.view_as(m.weight.data))

                elif isinstance(m, nn.Linear):
                    tmpbatchM = m.weight.data.t().clone()

                    tmpU, tmpS, tmpV = torch.svd(tmpbatchM)

                    for idx in range(0, tmpS.size(0)):
                        if tmpS[idx] > (1+self.svb_factor):
                            tmpS[idx] = 1+self.svb_factor
                        elif tmpS[idx] < 1/(1+self.svb_factor):
                            tmpS[idx] = 1/(1+self.svb_factor)

                    tmpbatchM = torch.mm(torch.mm(tmpU, torch.diag(tmpS.cuda())), tmpV.t()).t().contiguous()
                    m.weight.data.copy_(tmpbatchM.view_as(m.weight.data))
            # bbn
            if self.bbn_flag == True:
                if isinstance(m, nn.BatchNorm2d):
                    tmpbatchM = m.weight.data
                    if self.bbn_type == 'abs':
                        for idx in range(0, tmpbatchM.size(0)):
                            if tmpbatchM[idx] > (1+self.bbn_factor):
                                tmpbatchM[idx] = (1+self.bbn_factor)
                            elif tmpbatchM[idx] < 1/(1+self.bbn_factor):
                                tmpbatchM[idx] = 1/(1+self.bbn_factor)
                    elif self.bbn_type == 'rel':
                        mean = torch.mean(tmpbatchM)
                        relVec = torch.div(tmpbatchM, mean)
                        for idx in range(0, tmpbatchM.size(0)):
                            if relVec[idx] > (1+self.bbn_factor):
                                tmpbatchM[idx] = mean * (1+self.bbn_factor)
                            elif relVec[idx] < 1/(1+self.bbn_factor):
                                tmpbatchM[idx] = mean/(1+self.bbn_factor)
                    elif self.bbn_type == 'bbn':
                        running_var = m.running_var
                        eps = m.eps
                        running_std = torch.sqrt(torch.add(running_var, eps))
                        mean = torch.mean(tmpbatchM/running_std)
                        for idx in range(0, tmpbatchM.size(0)):
                            if tmpbatchM[idx]/(running_std[idx]*mean) > 1+self.bbn_factor:
                                tmpbatchM[idx] = running_std[idx] * mean * (1+self.bbn_factor)
                            elif tmpbatchM[idx]/(running_std[idx]*mean) < 1/(1+self.bbn_factor):
                                tmpbatchM[idx] = running_std[idx] * mean / (1+self.bbn_factor)

                    m.weight.data.copy_(tmpbatchM)

