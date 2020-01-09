import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class CrossEntropyLossWithOrtho(_WeightedLoss):

  def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, method = 'FNorm'):
    super(CrossEntropyLossWithOrtho, self).__init__(weight, size_average)
    self.ignore_index = ignore_index
    self.reduce = reduce
    self.method = method

  def forward(self, input, target, model, lemda):
    _assert_no_grad(target)
    extaFnorm = 0

    for m in model.modules():
      if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
        tmpM = m.weight.view(m.weight.data.size(0), -1).t()
        if self.method == 'FNorm':
          M = torch.mm(tmpM.t(), tmpM)
          extaFnorm +=  F.pairwise_distance(M, torch.eye(M.size(0)).cuda(), p=2).sum()
        elif self.method == 'SRIP':
          M = torch.mm(tmpM.t(), tmpM)
          tmp_extaFnorm =  M - torch.eye(M.size(0)).cuda()

          uni_nosie = torch.randn(tmpM.size(1),1).cuda()

          u = torch.mm(tmp_extaFnorm, uni_nosie)
          v = torch.mm(tmp_extaFnorm, u)

          extaFnorm += v.norm(p=2).sum() / (u.norm(p=2).sum() + 1e-12)

    loss = F.cross_entropy(input, target, self.weight, reduction='elementwise_mean') + lemda/2 * extaFnorm

    return loss