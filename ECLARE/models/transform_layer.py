from models.custom_embeddings import CustomEmbedding
from libs.utils import resolve_schema_args, fetch_json
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch


class scaled_spectral(nn.Module):
    def __init__(self, hidd_dims, k=1):
        super(scaled_spectral, self).__init__()
        self.layer = nn.utils.spectral_norm(nn.Linear(hidd_dims, hidd_dims))
        self.k = Parameter(torch.Tensor(1, 1))
        self.k.data.fill_(k)
        self.init()

    def forward(self, input):
        return self.k*self.layer(input)

    def extra_repr(self):
        return "spectral_radius = {}".format(self.k.data.numpy().shape)

    @property
    def stats(self):
        return "%f" % (self.k.detach().cpu().numpy()[0, 0])

    def init(self):
        nn.init.eye_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)


class spectral_attention(nn.Module):
    def __init__(self, input_dims, degree, norm="softmax"):
        super(spectral_attention, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.trans = nn.utils.spectral_norm(nn.Linear(input_dims, input_dims, True))
        self.layer = nn.utils.spectral_norm(nn.Linear(input_dims*degree, degree, False))
        self.nl = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.init()

    def attented(self, input, debug=False):
        _input = self.trans(self.nl(self.drop(input)))
        logits = self.layer(_input.flatten(start_dim=1))
        attn_wts = self.softmax(logits.squeeze())
        # if debug:
        #     wts = attn_wts.detach().unsqueeze(1)
        #     v = wts[:, :,:2].bmm(input[:,:2,:]).squeeze()
        #     u = wts.bmm(input).squeeze() - v
        #     print(v.shape, u.shape)
        #     v = torch.norm(v.detach(), dim=1).squeeze().numpy()
        #     u = torch.norm(u.detach(), dim=1).squeeze().numpy()
        #     weights = np.vstack([v, u]).T
        #     np.save("norm.npy", weights)
        #     print(np.mean(weights, axis=0))
        #     print(attn_wts.mean(axis=0).detach().cpu().numpy())
        return attn_wts

    def forward(self, input, debug=False):
        return self.attented(input, debug).unsqueeze(1).bmm(input).squeeze()

    def init(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.xavier_uniform_(self.trans.weight)
        self.trans.bias.data.fill_(0)


class coff(nn.Module):
    def __init__(self, input_dims, fill_val=1, nl=None):
        super(coff, self).__init__()
        self.k = Parameter(torch.Tensor(1, input_dims))
        self.k.data.fill_(fill_val)
        self.nl = nn.Identity()
        if nl == "sigmoid":
            self.nl = nn.Sigmoid()
        elif nl == "tanh":
            self.nl = nn.Tanh()

    def forward(self, input):
        return self.nl(self.k)*input

    def extra_repr(self):
        return "coff = {}".format(self.nl(self.k).data.numpy().shape)

    @property
    def stats(self):
        return "%0.2f" % (self.nl(self.k).detach().mean().cpu().numpy())


class Rpp(nn.Module):
    def __init__(self, input_size, output_size, dropout, nonLin="r", k=1):
        super(Rpp, self).__init__()
        self.name = "SR"
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.nonLin = nonLin
        self.padding_size = self.output_size - self.input_size
        elements = []
        if self.nonLin == 'r':
            elements.append(nn.ReLU())
        elif self.nonLin == 'lr':
            elements.append(nn.LeakyReLU())
        else:
            pass
        if dropout > 0.0:
            elements.append(nn.Dropout(p=dropout))
        self.nonLin = nn.Sequential(*elements)
        self.scaling = scaled_spectral(self.input_size, k)

    def forward(self, embed):
        return self.scaling(self.nonLin(embed)) + embed

    @property
    def stats(self):
        name = self.name
        s = "{}(K={})".format(name, self.scaling.stats)
        return s


class CRpp(Rpp):
    def __init__(self, input_size, output_size, dropout, nonLin="r",
                 k=1, non_linearity="sigmoid", fill_val=0):
        super(CRpp, self).__init__(input_size, output_size,
                                   dropout, nonLin, k)
        self.a = coff(input_size, fill_val, non_linearity)

    def forward(self, embed):
        return self.a(super(CRpp, self).forward(embed))

    @property
    def stats(self):
        name = self.name
        s = "{}(A={}, K={})".format(
            name, self.a.stats, self.scaling.stats)
        return s


elements = {
    'relu': nn.ReLU,
    'R': Rpp,
    'cR': CRpp,
    'BoW': CustomEmbedding,
    "light": coff,
    "dropout": nn.Dropout
}


class Transform(nn.Module):
    """
    Transform document representation!
    transform_string: string for sequential pipeline
        eg relu#,dropout#p:0.1,residual#input_size:300-output_size:300
    params: dictionary like object for default params
        eg {emb_size:300}
    """

    def __init__(self, modules, device="cuda:0"):
        super(Transform, self).__init__()
        self.device = device
        self.transform = nn.Sequential(*modules)

    def forward(self, embed):
        """
            Forward pass for transform layer
            Args:
                embed: torch.Tensor: document representation
            Returns:
                embed: torch.Tensor: transformed document representation
        """
        return self.transform(embed)

    def to(self):
        super().to(self.device)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            for module in super().__getattr__('transform'):
                try:
                    return module.__getattribute__(attr)
                except AttributeError:
                    return module.__getattr__(attr)
        raise AttributeError("{} not found".format(attr))

def get_functions(obj, params=None):
    return list(map(lambda x: elements[x](**obj[x]), obj['order']))
