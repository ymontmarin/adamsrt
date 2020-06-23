from functools import reduce

import torch
from torch.optim.optimizer import Optimizer


class SGDMRT(Optimizer):
    """
    General version of SGD-MRT that works for different normalization layer
    if specific channel options (channel_dims, channel_wise, channel_gloabal)
    are given.
    It should be used on parameters that are subject to scale invariance
    because they are followed by a normalization layer.
    Because not all params are concern, group_parameters of pytorch
    should be used.
    The effect is to RT transform the momentum.

    Example:
        >>> par_groups = [{'params': model.conv_params(), 'channel_wise'=True},
        >>>               {'params': model.other_params()}]
        >>> optimizer = SGDMRT(par_groups, lr=0.01, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Arguments:
        params (list of dict or iterator): either a list of group_parameters
            which is a dict with key 'params' with a param iterator and other
            key for optimizer parameters overiding for this group of param
        lr (float): learning rate
        momentum (float): momentum factor
        dampening (float): dampening in momentum calculation
        weight_decay (float): value for L2 regularization,
        channel_dims (list of int): the index of shape that represent the
            distinct dim that are independently normalized. Default value is
            channel_dims=shape which correspond to classic SGD.
            It can be used to adapt SGD to any normalization layers that
            follow conv layers
        channel_wise (bool): if True and channel_dims is None set it to [0]
            which correspond to classic channel shape in 2D conv Network.
            Normalization will be done over other dims which are subject to
            scale invariance thanks to following BN layer
    """
    def __init__(
        self,
        params,
        lr=1e-1,
        momentum=0,
        dampening=0,
        weight_decay=0,
        channel_dims=None,  # For customize the dimension for group of param
        channel_wise=False,  # True, For default conv followed by BN invariance
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            channel_dims=channel_dims,
            channel_wise=channel_wise,
        )
        super(SGDMRT, self).__init__(params, defaults)

    def step(self):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']

            for p in group['params']:
                # get grad
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0.:
                    grad.add_(p.data, alpha=weight_decay)

                # Get state
                state = self.state[p]

                # State initialization if needed
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    shape = p.data.shape
                    channel_dims = group['channel_dims']
                    if channel_dims is None:
                        if group['channel_wise']:
                            # Classic meaning of channels
                            channel_dims = [0]
                        else:
                            # element wise : every element is a channel
                            channel_dims = list(range(len(shape)))
                    state['channel_dims'] = channel_dims
                    state['shape'] = shape

                # Create the appropriate dot operator
                dot_ope = self.get_dot_operator(
                    state['channel_dims'], state['shape']
                )

                # Retrieve the buffer
                buf = state['momentum_buffer']

                # Update buffer
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                if dot_ope.dim > 1:
                    prev_data = p.data.clone().detach()

                p.data.add_(buf, alpha=-group['lr'])

                # We are on a sphere, we do RT transform
                if dot_ope.dim > 1:
                    new_data = p.data
                    new_norm_sq = dot_ope(new_data, new_data)
                    scal_x1_x2 = dot_ope(prev_data, new_data)
                    scal_m_x2 = dot_ope(buf, new_data)
                    # RT the order 1 moment
                    (
                        buf
                        .mul_(scal_x1_x2)
                        .add_(-scal_m_x2 * prev_data)
                        .div_(new_norm_sq)
                    )

    @staticmethod
    def get_dot_operator(channel_dims, shape):
        """
        Generate a function that do scalar product for each channel dims
        Over the remaining dims
        """
        # Other dims are the ones of groups of elem for each channel
        grp_dims = list(set(range(len(shape))) - set(channel_dims))

        # Compute shape and size
        channel_shape = [shape[i] for i in channel_dims]
        grp_shape = [shape[i] for i in grp_dims]
        channel_size = reduce(lambda x, y: x * y, [1] + channel_shape)
        grp_size = reduce(lambda x, y: x * y, [1] + grp_shape)

        # Prepare the permutation to ordonate dims and its reciproc
        perm = channel_dims + grp_dims
        antiperm = [
            e[1]
            for e in sorted([(j, i) for i, j in enumerate(perm)])
        ]

        # Prepare index query that retrieve all dimensions
        slice_len = max(len(channel_shape), 1)
        idx = [slice(None)] * slice_len + [None] * (len(shape) - slice_len)

        # Define the scalar product channel wise over grp dims
        # Output have is extend to fit initial shape
        def scalar_product(tensor1, tensor2):
            return (
                (tensor1 * tensor2)
                .permute(perm)  # permute as chan_dims, grp_dims
                .contiguous()
                .view(channel_size, grp_size)  # view as 2 dims tensor
                .sum(dim=1)  # norm over group dims to have scalar
                .view(*(channel_shape if channel_shape else [-1]))
                [idx]  # restore channel shape and extend on grp dims
                .permute(antiperm)  # Reverse permute to retrieve shape
                .contiguous()
            )
        # Add the dim of space of scalar product as attribute of the op
        scalar_product.dim = grp_size
        return scalar_product
