import torch
import torch.nn as nn

class Optimizer(object):
    """Wrapper for pytorch optimizer class to handle
    mixture of sparse and dense parameters
    Parameters
    ----------
    opt_type: str, optional, default='Adam'
        optimizer to use
    learning_rate: float, optional, default=0.01
        learning rate for the optimizer
    momentum: float, optional, default=0.9
        momentum (valid for SGD only)
    weight_decay: float, optional, default=0.0
        l2-regularization cofficient
    nesterov: boolean, optional, default=True
        Use nesterov method (useful in SGD only)
    freeze_embeddings: boolean, optional, default=False
        Don't update embedding layer
    """

    def __init__(self, opt_type='Adam', learning_rate=0.01,
                 momentum=0.9, weight_decay=0.0, nesterov=True,
                 freeze_embeddings=False):
        self.opt_type = opt_type
        self.optimizer = []
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.freeze_embeddings = freeze_embeddings

    def _get_opt(self, params, is_sparse):
        if self.opt_type == 'SGD':
            if is_sparse:
                return SparseSGD(
                    params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                )
            else:
                return torch.optim.SGD(
                    params,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )
        elif self.opt_type == 'Adam':
            if is_sparse:
                return torch.optim.SparseAdam(params)
            else:
                return torch.optim.Adam(params)
        else:
            raise NotImplementedError("Unknown optimizer!")

    def construct(self, model, lr=0.01):
        """
            Get optimizer.
            Args:
                model: torch.nn.Module: network
                params: : parameters
                freeze_embeddings: boolean: specify if embeddings need to be trained
            Returns:
                optimizer: torch.optim: optimizer as per given specifications  
        """
        model_params, is_sparse = self.get_params(model, lr)
        self.clear()
        self.optimizer = []
        for _, item in enumerate(zip(model_params, is_sparse)):
            if item[0]:
                self.optimizer.append(self._get_opt(
                    params=item[0], is_sparse=item[1]))
            else:
                self.optimizer.append(None)

    def adjust_lr(self, dlr_factor):
        """
            Adjust learning rate
            Args:
                dlr_factor: float: dynamic learning rate factor
        """
        for opt in self.optimizer:
            if opt:
                for param_group in opt.param_groups:
                    param_group['lr'] *= dlr_factor
                    if 'weight_decay' in param_group.keys():
                        param_group['weight_decay'] *= dlr_factor
    
    def clear(self):
        for opt in self.optimizer:
            if opt:
                del opt

    def step(self):
        for opt in self.optimizer:
            if opt:
                opt.step()
    
    def zero_grad(self):
        for opt in self.optimizer:
            if opt:
                opt.zero_grad()

    def load_state_dict(self, sd):
        for idx, item in enumerate(sd):
            if item:
                self.optimizer[idx].load_state_dict(item)

    def state_dict(self):
        out_states = []
        for item in self.optimizer:
            if item:
                out_states.append(item.state_dict())
            else:
                out_states.append(None)
        return out_states

    def get_params(self, net, args):
        self.net_params = {}
        self.net_params['sparse'] = list()
        self.net_params['dense'] = list()
        self.net_params['no_grad'] = list()
        if self.freeze_embeddings:
            for params in net.embed.parameters():
                params.requires_grad = False
        module_dict = net.__dict__['_modules']
        for m_name, val in module_dict.items():
            is_sparse = val.__dict__.get("sparse", False)
            is_decay = val.__dict__.get("weight_decay", False)
            lr = self.learning_rate
            for p_name, params in val.named_parameters():
                _decay, key = 0, "dense"
                if list(params.size())[0] > 1000:
                    _decay = self.weight_decay if is_decay else 0
                if params.requires_grad:
                    if is_sparse and list(params.size())[0] > 1000:
                        key = "sparse"
                    self.net_params[key].append(
                        {"params": params, "lr": lr, "weight_decay": _decay})
                else:
                    self.net_params['no_grad'].append(params)
        return [self.net_params['sparse'], self.net_params['dense']], \
            [True, False]
