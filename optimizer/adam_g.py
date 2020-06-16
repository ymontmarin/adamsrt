import math

import torch
from torch.optim.optimizer import Optimizer


class AdamG(Optimizer):
    """
    General version of Adam-SRT that works for different normalization layer
    if specific channel options (channel_dims, channel_wise, channel_gloabal)
    are given.
    It should be used on parameters that are subject to scale invariance
    because they are followed by a normalization layer.
    Because not all params are concern, group_parameters of pytorch
    should be used.
    The effect is to adapt moments of Adam to the geometry implied by
    normalization layer. RT transform the order one moment ; make the
    order 2 moment rescaled and by norm.

    Example:
        >>> par_groups = [{'params': model.conv_params(), 'channel_wise'=True},
        >>>               {'params': model.other_params()}]
        >>> optimizer = AdamSRT(par_groups, lr=0.01, betas=(0.9, 0.9999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Arguments:
        params (list of dict or iterator): either a list of group_parameters
            which is a dict with key 'params' with a param iterator and other
            key for optimizer parameters overiding for this group of param,
        lr (float): learning rate,
        betas (tuple(float)): momentum factor for Adam moments,
        eps (float): float to avoid numerical instality in normalization,
        weight_decay (float): value for L2 regularization,
        channel_dims (list of int): the index of shape that represent the
            distinct dim that are independently normalized. Default value is
            channel_dims=shape which correspond to classic Adam.
            It can be used to adapt Adam to any normalization layers that
            follow conv layers,
        channel_wise (bool): if True and channel_dims is None set it to [0]
            which correspond to classic channel shape in 2D conv Network.
            Normalization will be done over other dims which are subject to
            scale invariance thanks to following normalization layer,
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        channel_wise=False,  # For default conv followed by BN invariance
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            channel_wise=channel_wise,
        )
        super(AdamG, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimizatino step
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['channel_wise']:
                    # This is classic adam case
                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'])
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:
                    shape = grad.shape
                    square_grad = (
                        (grad * grad)
                        .view(shape[0], -1)
                        .sum(dim=1)
                        [[slice(None)] + [None] * (len(shape) - 1)]
                    )

                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, square_grad)

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1

                    previous_weight = p.data.clone().detach()
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                    # Transport momentum
                    new_weight = p.data
                    previous_norm_sq = (
                        (previous_weight * previous_weight)
                        .view(shape[0], -1)
                        .sum(dim=1)
                        [[slice(None)] + [None] * (len(shape) - 1)]
                    )
                    new_norm_sq = (
                        (new_weight * new_weight)
                        .view(shape[0], -1)
                        .sum(dim=1)
                        [[slice(None)] + [None] * (len(shape) - 1)]
                    )
                    scal_x1_x2 = (
                        (new_weight * previous_weight)
                        .view(shape[0], -1)
                        .sum(dim=1)
                        [[slice(None)] + [None] * (len(shape) - 1)]
                    )
                    scal_m_x2 = (
                        (new_weight * exp_avg)
                        .view(shape[0], -1)
                        .sum(dim=1)
                        [[slice(None)] + [None] * (len(shape) - 1)]
                    )
                    denom = (previous_norm_sq * new_norm_sq).sqrt()
                    (
                        exp_avg
                        .mul_(scal_x1_x2)
                        .add_(-scal_m_x2 * previous_weight)
                        .div_(denom.add_(group['eps']))
                    )

                    # Normalize new weights
                    p.data.div_(new_norm_sq.sqrt().add_(group['eps']))
