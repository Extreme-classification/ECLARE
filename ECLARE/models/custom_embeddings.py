import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CustomEmbedding(torch.nn.Module):
    """
        Memory efficient way to compute weighted EmbeddingBag
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, device="cuda:0"):
        """
            Args:
                num_embeddings: int: vocalubary size
                embedding_dim: int: dimension for embeddings
                padding_idx: int: index for <PAD>; embedding is not updated
                max_norm: 
                norm_type: int: default: 2
                scale_grad_by_freq: boolean: True/False
                sparse: boolean: sparse or dense gradients
        """
        super(CustomEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        _offset = 0
        if self.padding_idx is not None:
            _offset = 1
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(
            num_embeddings+_offset, embedding_dim))
        self.sparse = sparse
        self.device = torch.device(device)
        self.reset_parameters()

    def reset_parameters(self):
        """
            Reset weights
        """
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def to(self, element=None):
        if element is None:
            super().to(self.device)
        else:
            return element.to(self.device)

    def forward(self, X, _div=False):
        """
            Forward pass for embedding layer
            Args:
                features: dt.TLong: (batch_size, max_features_in_a_batch)
                weights: torch.Tensor: (batch_size, max_features_in_a_batch)
                _div: boolean: weighted sum or weighted average.
            Returns:
                out: torch.Tensor: embedding for each sample (batch_size, embedding_dims)
        """
        out = F.embedding_bag(self.to(X['X_ind']), self.weight, self.to(X['X_ptr']),
                              self.max_norm, self.norm_type,
                              self.scale_grad_by_freq, 'sum', self.sparse,
                              self.to(X['X_wts']), False)
        return out

    def _init_(self, words):
        self.weight.data.copy_(words['weight'])

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}, {device}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
