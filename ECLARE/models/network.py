import math
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import models.transform_layer as tl
from xclib.utils import sparse as xs
from xclib.data import data_utils as du
import models.linear_layer as linear_layer
from sklearn.preprocessing import normalize
from libs.utils import normalize_graph, print_stats, fetch_json

class GraphBase(torch.nn.Module):
    def __init__(self, params):
        self._set_params(params)
        super(GraphBase, self).__init__()
        self.vocabulary_dims = params.vocabulary_dims+1
        self.embedding_dims = params.embedding_dims
        self.num_labels = params.num_labels
        self.embed = tl.get_functions(self.embeddings)[0]

    def _set_params(self, params):
        self.topk_edges = params.top_k
        self.labelfts_transform = None
        params.trans_config_lbl = None
        transform_config_dict = fetch_json(params.trans_method, params)
        self.embeddings = transform_config_dict['embedding']
        self.transform = transform_config_dict['trans_emb']
        self.lblfts_transform = transform_config_dict['trans_gph']
        self.to_clusters = None
        self.filters = None
        self.degree = params.degree
        self.fixed_lbs = params.freeze_embeddings
        self.lbs_params = None
        self.offset = 0

    def _construct_classifier(self):
        pass

    def _construct_transform(self, trans_config):
        return tl.Transform(tl.get_functions(trans_config))

    @property
    def representation_dims(self):
        return self.embed.embedding_dim

    def encode(self, batch_data, return_course=False):
        """encode documents
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features

        Returns
        -------
        out: torch.Tensor
            encoding of a document
        """
        return self.embed(batch_data)

    def forward(self, batch_data):
        """Forward pass
        Parameters:
        -----------
        batch_data: dict
            batch_data['X']: torch.Tensor
                dense feature vector or
                feature indices in case of sparse features
            batch_data['X_w]: torch.Tensor
                feature weights in case of sparse features
        Returns
        -------
        out: logits for each label
        """
        return self.encode(batch_data)

    def to(self):
        self.embed.to()
        pass

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def model_size(self):
        return self.num_trainable_params * 4 / math.pow(2, 20)

    def _init_(self, words, **kwargs):
        self.embed._init_(words)


class ECLAREh(GraphBase):
    def __init__(self, params):
        super(ECLAREh, self).__init__(params)
        self._sparse_matrix(params)
        self.clf_from_fts, self.clf_from_wts = self._construct_classifier(
            params)
        self.transform = self._construct_transform(self.transform)

    def encode(self, batch_data, return_course=False):
        if batch_data['is_sparse']:
            embed = super(ECLAREh, self).encode(batch_data)
        else:
            embed = batch_data['X'].cuda()
        if return_course:
            return embed
        return self.transform(embed)

    def _create_mapping(self, clusters, num_lbs):
        to_cluster = sp.lil_matrix((len(clusters), num_lbs), dtype=np.float)
        rows = np.concatenate(list(
            map(lambda x: np.tile(x[0], x[1].size),
                enumerate(clusters))))
        cols = np.concatenate(clusters)
        to_cluster[rows, cols] = 1
        return to_cluster.tocsr()

    def _padded(self, mat, new_shape):
        mat = mat.tocoo()
        mat = sp.csr_matrix((mat.data, (mat.row, mat.col)), shape=new_shape)
        mat.sort_indices()
        return mat

    def _load_lblft(self, params, return_padded=False):
        label_word = params.lblft
        if self.to_clusters is not None:
            label_word = self.to_clusters.dot(label_word)
            label_word = normalize(label_word, norm='l2')

        if return_padded:
            row, col = label_word.shape
            label_word = self._padded(label_word, (row+1, col))
        label_word.sort_indices()
        return label_word.tocsr()

    def _load_graph(self, params, return_padded=False, train=True):
        graph = params.graph.copy()

        if self.to_clusters is not None:
            graph = self.to_clusters.dot(graph)
            graph = graph.dot(self.to_clusters.transpose()).tocsr()
            print("Squeezing Graph", self.topk_edges)
            graph = xs.retain_topk(graph, k=self.topk_edges)
            print_stats(graph)
        graph = normalize_graph(graph)

        if return_padded:
            row, col = graph.shape
            graph = self._padded(graph, (row+1, col+1))
        graph.sort_indices()
        return graph.tocsr()

    def _set_params(self, params):
        super(ECLAREh, self)._set_params(params)
        self.use_classifier_wts = params.use_classifier_wts
        num_lbs = params.lblft.shape[0]
        if params.clusters is not None:
            self.to_clusters = self._create_mapping(params.clusters, num_lbs)
        self.label_padding_idx = None
        self.sparse_clf = False
        self.sparse_emb = False
        self.sparse_gph = False
        self.graph_train = None
        self.use_bias = False
        self.fixed_lbs = params.freeze_embeddings
        self.lbs_params = None
        self.offset = 0
        _order0 = self.embeddings['order'][0]
        self.embeddings[_order0]['sparse'] = self.sparse_emb

    def _BSMatComp(self, index, values, shape):
        mat = torch.sparse_coo_tensor(index, values, shape)
        mat._coalesced_(True)
        return mat

    def _BSMat(self, mat):
        mat = mat.tocsr()
        mat.sort_indices()
        mat = mat.tocoo()
        values = torch.FloatTensor(mat.data)
        index = torch.LongTensor(np.vstack([mat.row, mat.col]))
        shape = torch.Size(mat.shape)
        return self._BSMatComp(index, values, shape)

    def forward(self, batch_data):
        input = self.encode(batch_data)
        label_fts = self._get_lbl_fts()
        return self.clf_from_wts(input, label_fts, None, None)

    def _predict(self, batch_data):
        input = self.encode(batch_data)
        return self.clf_from_wts(input, None, None)

    def _traverse(self, smat):
        return smat.dot(self.graph_pred)

    def get_weights(self):
        clf0 = self.clf_from_fts.get_weights()
        clf1 = self.clf_from_wts.get_weights()
        return {'words': clf0, 'classifier': clf1}

    def _get_lbl_fts(self, *args):
        if self.fixed_lbs and self.lbs_params is None:
            self._setup_graph(not self._device == 'cpu')
        if self.fixed_lbs:
            return self.clf_from_fts(self.lbs_params, None)
        return self.clf_from_fts(self.label_word, self.graph_train,
                                 weights=self.embed.weight)

    def _pred_init(self):
        self.clf_from_wts._setup(self._get_lbl_fts())
        if self.clf_from_wts is not None:
            print(self.clf_from_wts.stats)
        if self.clf_from_fts is not None:
            print(self.clf_from_fts.stats)

    def _eval(self):
        self.eval()
        self._pred_init()

    def _train(self):
        self.clf_from_wts.clean()
        self.train()

    def _setup_graph(self, to_cuda=True):
        if not self.fixed_lbs:
            if to_cuda:
                self.label_word = self.label_word.cuda()
                self.graph_train = self.graph_train.cuda()
            else:
                self.label_word = self.label_word.cpu()
                self.graph_train = self.graph_train.cpu()
        else:
            if self.lbs_params is None:
                tokens = self.embed.weight.detach().cpu()
                self.lbs_params = [self.label_word.cpu().mm(tokens)]
                for _ in range(1, self.degree):
                    _params = self.graph_train.cpu().mm(self.lbs_params[-1])
                    self.lbs_params.append(_params)
            if to_cuda:
                for x in range(self.degree):
                    self.lbs_params[x] = self.lbs_params[x].cuda()
            else:
                for x in range(self.degree):
                    self.lbs_params[x] = self.lbs_params[x].cpu()

    def to(self):
        self._device = "cuda:0"
        self.embed.to()
        self.transform.to()
        self.clf_from_wts.to()
        self.clf_from_fts.to()
        self._setup_graph()

    def cpu(self):
        self._device = "cpu"
        self.embed = self.embed.cpu()
        self.transform = self.transform.cpu()
        self.clf_from_wts = self.clf_from_wts.cpu()
        self.clf_from_fts = self.clf_from_fts.cpu()
        self._setup_graph(to_cuda=False)

    def _construct_classifier(self, params):
        clf0, clf1 = None, None
        clf0 = linear_layer.GraphConv(
            input_size=self.representation_dims,
            label_features=params.label_features,
            sparse=self.sparse_gph, padding_idx=self.label_padding_idx,
            name="GCN", reduction='add', degree=self.degree,
            graph_layer=self.lblfts_transform
        )
        clf1 = linear_layer.GraphCombine(
            input_size=self.representation_dims,
            output_size=int(self.num_labels + self.offset),
            use_classifier_wts=params.use_classifier_wts,
            sparse=self.sparse_clf,
            padding_idx=self.label_padding_idx,
            degree=self.degree, bias=self.use_bias
        )
        return clf0, clf1

    def _sparse_matrix(self, params):
        self.label_word = self._BSMat(self._load_lblft(params))
        self.graph_train = self._BSMat(self._load_graph(params))
        self.graph_pred = self._load_graph(params, train=False)

    def _init_(self, words, clf):
        self.embed._init_(words)
        self.clf_from_fts._init_(clf)
        self.clf_from_wts._init_(clf, self.label_word, self.graph_train)


class ECLAREt(ECLAREh):
    def __init__(self, params):
        super(ECLAREt, self).__init__(params)

    def _set_params(self, params):
        super(ECLAREt, self)._set_params(params)
        self.sparse_clf = True
        self.sparse_emb = True
        self.use_bias = False
        self.offset = 1

    def _fixed_forward(self, batch_data):
        input = self.encode(batch_data)
        Y_r, shorty = batch_data['Y_r'].cuda(), batch_data['f_shoty'].cuda()
        data = list(map(lambda x: F.embedding(
            shorty, self.lbs_params[x], sparse=False), np.arange(self.degree)))
        label_fts = self._get_lbl_fts(data, None, None)
        return self.clf_from_wts(input, label_fts, Y_r, shorty)

    def _train_forward(self, batch_data):
        input = self.encode(batch_data)
        label_wrd = self._BSMatComp(*batch_data['shoty_lf']).cuda()
        graph_trn = self._BSMatComp(*batch_data['shoty_lg']).cuda()
        lbl_fts = batch_data['lf'].cuda()
        weights = F.embedding(lbl_fts, self.embed.weight, sparse=True)
        Y_r, shorty = batch_data['Y_r'].cuda(), batch_data['f_shoty'].cuda()

        label_fts = self._get_lbl_fts(label_wrd, graph_trn, weights)
        return self.clf_from_wts(input, label_fts, Y_r, shorty)

    def forward(self, batch_data):
        if self.fixed_lbs:
            return self._fixed_forward(batch_data)
        else:
            return self._train_forward(batch_data)

    def _predict(self, batch_data):
        input = self.encode(batch_data)
        return self.clf_from_wts(input, None, batch_data['Y_s'].cuda())

    def _get_lbl_fts(self, lbf=None, graph=None, wts=None):
        if self.fixed_lbs and self.lbs_params is None:
            self._setup_graph(not self._device == 'cpu')
        if self.fixed_lbs:
            if lbf is None:
                lbf = self.lbs_params
            return self.clf_from_fts(lbf, None)
        if graph is None:
            graph = self._BSMat(self.graph_train).to(self._device)
        if lbf is None:
            lbf = self._BSMat(self.label_word)
        if wts is None:
            wts = self.embed.weight
        return self.clf_from_fts(lbf, graph, weights=wts)

    def _setup_graph(self, to_cuda=True):
        if self.fixed_lbs:
            if self.lbs_params is None:
                print("Pre storing word embeddings")
                tokens = self.embed.weight.detach().cpu()
                lbl_params = self._BSMat(self.label_word).cpu()
                self.lbs_params = [lbl_params.mm(tokens)]
                graph = self._BSMat(self.graph_train).cpu()
                for _ in range(1, self.degree):
                    _params = graph.mm(self.lbs_params[-1])
                    self.lbs_params.append(_params)

            if to_cuda:
                for x in np.arange(self.degree):
                    self.lbs_params[x] = self.lbs_params[x].cuda()
            else:
                for x in np.arange(self.degree):
                    self.lbs_params[x] = self.lbs_params[x].cpu()
        pass

    def _sparse_matrix(self, params):
        self.label_word = self._load_lblft(params, True)
        self.graph_train = self._load_graph(params, True)
        self.graph_pred = self._load_graph(params, True, False)

    def _init_(self, words, clf):
        self.embed._init_(words)
        self.clf_from_fts._init_(clf)
        _label_wrds = self._BSMat(self.label_word).cpu()
        _label_gph = self._BSMat(self.graph_train).cpu()
        self.clf_from_wts._init_(clf, _label_wrds, _label_gph)
