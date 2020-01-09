import os
import shutil
import numpy as np
import time
from threading import Thread
import sys

import torch
import torchvision
from torch.autograd import Variable

last_time = time.time()
begin_time = last_time
'''
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    step_time = Average_meter()
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: {0}'.format(format_time(step_time)))
    L.append(' | Tot: {0}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
'''
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class Average_meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Training_aux(object):
    def __init__(self, fsave):
        self.fsave = fsave

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batchSize = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correctK = correct[:k].view(-1).float().sum(0)
            res.append(correctK.mul_(100.0 / batchSize))
        return res

    def save_checkpoint(self, state, is_best, filename):
        """Saves checkpoint to disk"""
        ''' 
        usage:
        Training_aux.save_checkpoint(
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best = is_best)
        '''
        directory = "%s/"%(self.fsave)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, '%s/'%(self.fsave) + 'modelBest.pth.tar')
            self.write_err_to_file(epoch = 0, top1 = 0, top5 = 0, trn_loss = 0, mode = 'best')
        return 

    def load_checkpoint(self, model, optimizer, is_best):
        """Loads checkpoint from disk"""
        ''' 
        usage:
        start_epoch, best_prec1 = Training_aux.load_checkpoint(model = model, is_best = is_best)
        '''
        directory = "%s/"%(self.fsave)
        if is_best:
            filename = directory + 'modelBest.pth.tar'
            print("=> loading best model '{}'".format(filename))
        else:
            filename = directory + 'checkpoint.pth.tar'
            print("=> loading checkpoint '{}'".format(filename))

        if os.path.isfile(filename):
            checkpoint = torch.load(filename)

            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            
            model.load_state_dict(checkpoint['state_dict'])

            print("==> loaded checkpoint '{}' (epoch {})"
                    .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return start_epoch, best_prec1

    def write_err_to_file(self, epoch, top1 = 0, top5 = 0, 
            adv_top1 = 0, adv_top5= 0, trn_loss = 0, 
            top1_mixup = 0, top5_mixup = 0, trnLoss_mixup = 0, 
            mode= 'train'):
        """write error to txt"""
        """mode ~ {'train' , 'val' , 'best'}"""
        fpath = self.fsave+'/state.txt'
        directory = "%s/"%(self.fsave)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.isfile(fpath):
            file = open(fpath, 'a')
        else:
            file = open(fpath, "w")
        
        if mode == 'train':
            file.write(
                        'Training-Epoch:{0}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(epoch, loss = trn_loss, top1 = top1, top5 = top5))
        elif mode == 'val':
             file.write( 
                        '\t\t\t\t\t\t\t\t\t\t\t\t'
                        'Testing-Epoch:{0}\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(epoch, top1 = top1, top5 = top5))   
        elif mode == 'loss_only':
             file.write( 
                        'Training-Epoch:{0}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, loss = trn_loss))  
        elif mode == 'best':
            file.write( '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t'
                        '***Best so far ***\n')
        else:
            raise Exception('The mode of writing erros to file must be either train / val / adv_val / train_mixup / best!')

        file.close()
        return 
