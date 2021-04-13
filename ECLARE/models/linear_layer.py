import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import models.transform_layer as tl
import math


def add_intend(str, num_intend=1):
    _s = str.split("\n")
    _s = _s[0] + "\n" + "\n".join(list(map(lambda x: "\t"+x, _s[1:])))
    return _s


class GraphConv(nn.Module):
    def __init__(self, input_size, label_features, padding_idx=None,
                 device="cuda:0", sparse=False, name=None, degree=2,
                 bias=False, graph_layer="", reduction='max'):
        super(GraphConv, self).__init__()
        self.device = device
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__
        self.input_size = input_size
        self.label_features = label_features
        self.sparse = sparse
        self._bias = bias
        self.padding_idx = padding_idx
        self.graph_layer = graph_layer
        self.degree = degree
        for idx in np.arange(self.degree):
            self.__setattr__('func_%d' % idx, tl.get_functions(graph_layer)[0])
        self.reset_parameters()

    def _learn(self, labels_fts, graph_lbl, weights):
        Ut = labels_fts.mm(weights)
        order_p = [self[0](Ut)]
        for idx in np.arange(1, self.degree):
            Ut = graph_lbl.mm(Ut)
            order_p.append(self[idx](Ut))
        return order_p

    def _fixed(self, fts):
        return list(map(lambda x: self[x](fts[x]), np.arange(self.degree)))

    def forward(self, labels_fts, graph_lbl, weights=None):
        if weights is None:
            return self._fixed(labels_fts)
        else:
            return self._learn(labels_fts, graph_lbl, weights)

    def to(self):
        """Transfer to device
        """
        super().to(self.device)

    def reset_parameters(self):
        """Initialize vectors
        """
        pass

    def extra_repr(self):
        s = "input={input_size}, output={label_features}, sparse={sparse}"
        return s.format(**self.__dict__)

    @property
    def stats(self):
        name = self.name
        s = []
        for idx in np.arange(self.degree):
            s.append(add_intend("a(%d) = %s" % (idx, self[idx].stats)))
        s = "{}:(\n{})".format(name, ''.join(s))
        return s
    
    def _init_(self, words):
        pass
        # if self.weight is not None:
        #     self.weight.data.copy_(words['weight'])

    def __getitem__(self, idx):
        return self.__getattr__('func_%d' % idx)


class GraphCombine(nn.Module):
    def __init__(self, input_size, output_size, bias=True, reduction='max',
                 use_classifier_wts=False, padding_idx=None, name='Linear',
                 device="cuda:0", sparse=False, degree=1):
        super(GraphCombine, self).__init__()
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__
        self.device = device  # Useful in case of multiple GPUs
        self.input_size = input_size
        self.output_size = output_size
        self.use_classifier_wts = use_classifier_wts
        self.sparse = sparse
        self.padding_idx = padding_idx
        self.weight_decay = False
        self.degree = degree
        if self.use_classifier_wts:
            if bias:
                if self.sparse:
                    self.bias = Parameter(torch.Tensor(self.output_size, 1))
                else:
                    self.bias = Parameter(torch.Tensor(self.output_size))
            else:
                self.register_parameter('bias', None)
            self.weight = Parameter(torch.Tensor(self.output_size,
                                                 self.input_size))
            self.degree += 1
        else:
            self.register_parameter('bias', None)
            self.register_parameter('weight', None)
        self.pre_build = [None, None]
        self.attention = tl.spectral_attention(self.input_size, self.degree)
        self.print = False
        self.reset_parameters()

    def _get_clf(self, wts, shorty=None, sparse=True, padding_idx=None):
        if shorty is not None and wts is not None:
            return F.embedding(shorty, wts, sparse=sparse,
                               padding_idx=padding_idx)
        return wts

    def _rebuild(self, lclf, shorty=None):
        if self.pre_build[0] is None:
            if self.use_classifier_wts:
                lclf.append(self._get_clf(self.weight, shorty,
                                          True, self.padding_idx))

            lclf = torch.stack(lclf, dim=1)
            lclf = self.attention(lclf, self.print)
            bias = self._get_clf(self.bias, shorty, True, self.padding_idx)
        else:
            lclf, bias = self.pre_build
        return lclf, bias

    def forward(self, input, lbl_ft, shorty=None, shorty_f=None):
        lclf, bias = self._rebuild(lbl_ft, shorty_f)
        if shorty is not None:
            lclf = self._get_clf(lclf, shorty, sparse=False)
            input = input.unsqueeze(1)
            out = input.bmm(lclf.permute(0, 2, 1)).squeeze()
            if bias is not None:
                bias = self._get_clf(bias, shorty, sparse=False).squeeze()
                out += bias
            return out.squeeze()
        else:
            return F.linear(input, lclf, bias)

    def _setup(self, lbl_ft):
        self.print = True
        lclf, bias = self._rebuild(lbl_ft, None)
        if bias is not None:
            bias = bias + 0
        self.pre_build = [lclf, bias]
        self.print = False

    def to(self):
        """Transfer to device
        """
        if self.pre_build[0] is not None:
            self.pre_build[0] = self.pre_build[0].to(self.device)
        if self.pre_build[1] is not None:
            self.pre_build[1] = self.pre_build[1].to(self.device)
        super().to(self.device)

    def cpu(self):
        if self.pre_build[0] is not None:
            self.pre_build[0] = self.pre_build[0].cpu()
        if self.pre_build[1] is not None:
            self.pre_build[1] = self.pre_build[1].cpu()
        return super().cpu()

    def reset_parameters(self):
        """Initialize vectors
        """
        stdv = 1. / math.sqrt(self.input_size)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.fill_(0)

    def get_wts(self):
        """Get wts as dictionary
        """
        parameters = {}
        if self.bias is not None:
            parameters['bias'] = self.bias.detach().cpu().numpy()
        if self.use_classifier_wts:
            parameters['weight'] = self.weight.detach().cpu().numpy()
        return parameters

    def extra_repr(self):
        s = "output_size={output_size}, sparse={sparse}"
        if self.use_classifier_wts:
            s += ", bias={}".format(str(self.bias is not None))
            s += ", weight={}".format((self.input_size, self.output_size))
        return s.format(**self.__dict__)

    @property
    def stats(self):
        return ""

    def clean(self):
        if self.pre_build[0] is not None:
            self.pre_build[0] = None
        if self.pre_build[1] is not None:
            self.pre_build[1] = None
    
    def _init_(self, words, lbl_wrds, graph):
        if self.use_classifier_wts:
            print("Initializing with precomputed features")
            embeddings = words['weight'].cpu()
            v0 = lbl_wrds.mm(embeddings)
            for i in np.arange(self.degree-2):
                print("graph:", i)
                v0 = graph.mm(v0)
            self.weight.data.copy_(v0)
