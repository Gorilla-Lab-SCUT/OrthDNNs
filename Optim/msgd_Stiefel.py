import torch
from .optimizer import Optimizer, required

class msgd_Stiefel(Optimizer):
    r"""
    sgd based optimization along Stiefel manifolds of weight matrices 
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(msgd_Stiefel, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(msgd_Stiefel, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if p.data.dim() == 4:
                    n_feature_out = p.data.size(0)
                    n_feature_in = p.data.size(1)
                    kernel_h = p.data.size(2)
                    kernel_w= p.data.size(3)

                    if kernel_h*kernel_w*n_feature_in <= n_feature_out:
                        tmp_I = torch.eye(n_feature_out).cuda()
                        W = p.data.reshape(n_feature_out, kernel_h*kernel_w*n_feature_in).clone()
                        dW = d_p.reshape(n_feature_out, kernel_h*kernel_w*n_feature_in).clone()
                    else:
                        tmp_I = torch.eye(kernel_h*kernel_w*n_feature_in).cuda()
                        W = p.data.reshape(n_feature_out, kernel_h*kernel_w*n_feature_in).t().clone()
                        dW = d_p.reshape(n_feature_out, kernel_h*kernel_w*n_feature_in).t().clone()

                    W_new = W - group['lr'].cuda()*(torch.mm((tmp_I - torch.mm(W, W.t())),dW) + 0.5*torch.mm(W, (torch.mm(W.t(), dW) - torch.mm(dW.t(), W))))
                    Q, R = torch.qr(W_new)  #retraction onto the orthogonal manifold, THE DOMINATING COMPUTATION STEP
                    D = torch.diag(torch.sign(torch.diag(R))) 
                    Q = torch.mm(Q,D)   #change sign to keep the retraction Q unique, since QR factorization are unique only up to sign change

                    if kernel_h*kernel_w*n_feature_in <= n_feature_out:
                        p.data = (Q.reshape(n_feature_out, n_feature_in, kernel_h, kernel_w)).clone()
                    else:
                        p.data = (Q.t().reshape(n_feature_out, n_feature_in, kernel_h, kernel_w)).clone()

                elif p.data.dim() == 2:
                    n_feature_out = p.data.size(0)
                    n_feature_in = p.data.size(1)
                    if n_feature_in <= n_feature_out:
                        tmp_I = torch.eye(n_feature_out).cuda()
                        W = p.data.clone()
                        dW = d_p.clone()
                    else:
                        tmp_I = torch.eye(n_feature_in).cuda()
                        W = p.data.t().clone()
                        dW = d_p.t().clone()

                    W_new = W - group['lr'].cuda()*(torch.mm((tmp_I - torch.mm(W, W.t())),dW) + 0.5*torch.mm(W, (torch.mm(W.t(), dW) - torch.mm(dW.t(), W))))
                    Q, R = torch.qr(W_new)
                    D = torch.diag(torch.sign(torch.diag(R)))
                    Q = torch.mm(Q,D)

                    if n_feature_in <= n_feature_out:
                        p.data = Q
                    else:
                        p.data = Q.t()
                else:   #for bias vectors of conv, fc, and BN layers, and weight vector of BN layer 
                    p.data.add_(-group['lr'], d_p)

        return loss