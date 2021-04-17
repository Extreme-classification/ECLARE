import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import models.transform_layer as transform_layer
from libs.utils import normalize_graph, PrunedWalk
from sklearn.preprocessing import normalize
from xclib.utils.sparse import retain_topk
from libs.tree import build_tree as Tree
import xclib.evaluation.xc_metrics as xc
from .network import *
import math
import copy
import os



class ECLAREBase(nn.Module):
    def __init__(self, params):
        super(ECLAREBase, self).__init__()
        self.params = params
        self.lblft_npz = self._load_lblft(params)
        self.graph_npz = self._load_graph(params)
        self.Xs = None
        self.degree = params.degree
        self.tree = Tree(b_factors=params.b_factors,
                         method=params.cluster_method,
                         leaf_size=params.num_labels,
                         force_shallow=params.force_shallow)
        self.m_size = 0
        self.depth = -1
        self.depth_node = nn.Sequential()

    def _load_lblft(self, params):
        v_lbl = np.loadtxt(params.label_indices, dtype=np.int)
        label_features = np.loadtxt(params.v_lbl_fts, dtype=np.int)
        label_word = os.path.join(
            params.data_dir, params.dataset, params.label_words)
        label_word = du.read_sparse_file(label_word)[v_lbl]
        label_word = label_word.tocsc()[:, label_features]
        padd = sp.lil_matrix((v_lbl.size, 1))
        label_word = sp.hstack([label_word, padd]).tocsr()
        label_word = normalize(label_word, norm='l2')
        label_word.sort_indices()
        return label_word.tocsr()

    def _load_graph(self, params):
        print(os.path.join(params.model_dir, params.graph_name))
        if not os.path.exists(os.path.join(
                params.model_dir, params.graph_name)):
            trn_y = du.read_sparse_file(
                os.path.join(params.data_dir, params.dataset,
                             params.tr_label_fname))
            valid_lbs = np.loadtxt(params.label_indices, dtype=int)
            trn_y = trn_y.tocsc()[:, valid_lbs]

            n_lbs = valid_lbs.size
            diag = np.ones(n_lbs, dtype=np.int)

            if params.verbose > 0:
                verbose_labels = np.where(
                    np.ravel(trn_y.sum(axis=0) > params.verbose))[0]
                print("Verbose_labels:", verbose_labels.size)
                diag[verbose_labels] = 0
            else:
                verbose_labels = np.asarray([])
            diag = sp.diags(diag, shape=(n_lbs, n_lbs))
            print("Avg: labels", trn_y.nnz/trn_y.shape[0])
            trn_y = trn_y.dot(diag).tocsr()
            trn_y.eliminate_zeros()
            yf = None
            if os.path.exists(params.embeddings):
                emb = torch.load(params.embeddings)['weight'].cpu().numpy()
                yf = normalize(self.lblft_npz.dot(emb))
            graph = PrunedWalk(trn_y, yf=yf).simulate(
                params.walk_len, params.p_reset,
                params.top_k, max_dist=params.prune_max_dist)
            if verbose_labels.size > 0:
                graph = graph.tolil()
                graph[verbose_labels, verbose_labels] = 1
                graph = graph.tocsr()
            sp.save_npz(os.path.join(
                params.model_dir, params.graph_name), graph)
        else:
            graph = sp.load_npz(os.path.join(
                params.model_dir, params.graph_name))
        return graph

    def build(self, lbl_cnt=None, label_idx=None, verbose_lbs=None, model_dir=None):
        n_gph = normalize_graph(self.graph_npz)
        if not os.path.exists(model_dir+"/clusters.pkl"):
            if self.params.cluster_method == 'AugParabel':
                print("Augmenting graphs")
                if isinstance(lbl_cnt, np.ndarray):
                    lbl_cnt = n_gph.dot(normalize(lbl_cnt))
                elif sp.issparse(lbl_cnt):
                    print("Avg features", lbl_cnt.nnz / lbl_cnt.shape[0])
                    lbl_cnt = n_gph.dot(lbl_cnt).tocsr()
                    lbl_cnt = retain_topk(lbl_cnt.tocsr(), k=1000).tocsr()
                    print("Avg features", lbl_cnt.nnz / lbl_cnt.shape[0])
                else:
                    print("Do not understand the type")
                    exit(0)
            self.tree.fit(label_idx, verbose_lbs, lbl_cnt)
            self._setup()
            self.save(model_dir)
        else:
            self.load(model_dir)

    def _setup(self):
        self.C = self.tree.C
        self.height = self.tree.height
        self.leaf_hash = self.tree.hash_map_array[-1]._get_hash()
        self.leaf_weit = self.tree.hash_map_array[-1]._get_weights()

    def _layer(self, params, model):
        if model == "ECLAREh":
            return ECLAREh(params)
        elif model == "ECLAREt":
            return ECLAREt(params)
        elif model == "Base":
            return GraphBase(params)
        else:
            print("{}:Kuch bhi".format(self.__class__.__name__))

    def _add_layer(self, depth):
        hash_map_obj = self.tree.hash_map_array[depth]
        params = self.params
        params.num_labels = hash_map_obj.size
        params.clusters = hash_map_obj.clusters
        params.prev_cluster = self.tree.hash_map_array[depth-1].clusters
        params.label_padding_index = hash_map_obj.padding_idx
        params.lblft = self.lblft_npz
        params.graph = self.graph_npz
        for _params in self.depth_node.parameters():
            _params.requires_grad = False
        self.depth_node = self._layer(params, params.layers[depth])
        for params in self.depth_node.parameters():
            self.m_size += params.numel()

    def _get_encoder(self):
        encoder = self._layer(self.params, "Base")
        encoder._init_(self.word_wts)
        return encoder

    def scores(self, batch, depth=0):
        return self.depth_node(batch)

    def forward(self, batch, depth=None):
        if depth is None:
            depth = self.depth
        scores = self.scores(batch, depth=depth)
        return scores

    def _predict(self, batch):
        return F.logsigmoid(self.depth_node._predict(batch))

    def traverse_graph(self, smat, depth=0):
        return self.depth_node._traverse(smat)

    def _clusters(self, depth=None):
        return self.tree._get_cluster_depth(depth)

    def _prep_for_depth(self, depth, initialize=True, disable_grad=True):
        if initialize:
            self._initialize_next_layer(depth)
        self._add_layer(depth)
        if initialize:
            self.depth_node._init_(self.word_wts, self.word_wts)
        self._call_back()
        self.depth = depth

    def _build_predict(self, dataloader, depth):
        pass

    def eval(self, depth=None):
        self.depth_node._eval()

    def train(self):
        self.depth_node.to()
        self.depth_node._train()
        self._call_back()

    def initialize_embeddings(self, weights):
        self.word_wts = weights

    def init_clf_embeddings(self, weights):
        pass

    def to(self):
        self.depth_node.to()

    def cpu(self):
        self.depth_node.cpu()

    def _call_back(self):
        pass

    def _initialize_next_layer(self, depth):
        pass

    @property
    def label_words(self):
        """
        Designed to be used only with k>1
        """
        if self.params.layers[self.depth] in ["ECLAREt"]:
            return self.depth_node.label_word
        return None

    @property
    def graph(self):
        """
        Designed to be used only with k>1
        """
        if self.params.layers[self.depth] in ["ECLAREt"]:
            gph = self.depth_node.graph_pred
            for _ in np.arange(1, self.degree-1):
                gph = gph.dot(gph)
            return gph.tocsr()
        return None

    @property
    def model_size(self):
        return self.m_size*4/np.power(2, 20)

    def get_weights(self):
        clf_label_fts = {}
        for depth in range(self.height):
            clf_label_fts[str(depth)] = self[0].get_weights()
        return clf_label_fts

    def save(self, fname):
        fname = fname + "/clusters.pkl"
        self.tree.save(fname)

    def load(self, fname):
        fname = fname + "/clusters.pkl"
        self.tree.load(fname)
        self._setup()

    @property
    def offset(self):
        return self.depth_node.offset

    @property
    def isleaf(self):
        return self.depth == self.tree.height-1


class ECLAREpp(ECLAREBase):
    """docstring for treeClassifier"""

    def __init__(self, params):
        super(ECLAREpp, self).__init__(params)

    def _initialize_next_layer(self, depth):
        if depth > 0:
            print("Setting from top layer")
            self.word_wts = self.depth_node.embed.state_dict()


class ECLAREfe(ECLAREBase):
    """docstring for treeClassifier"""

    def __init__(self, params):
        super(ECLAREfe, self).__init__(params)

    def _initialize_next_layer(self, depth):
        pass

    def build(self, lbl_cnt=None, label_idx=None, verbose_lbs=None, model_dir=None):
        n_gph = normalize_graph(self.graph_npz)
        if not os.path.exists(model_dir+"/clusters.pkl"):
            if self.params.cluster_method == 'AugParabel':
                lbl_cnt = normalize(n_gph.dot(lbl_cnt))
            self.tree.fit(label_idx, verbose_lbs, lbl_cnt)
            self._setup()
            self.save(model_dir)
        else:
            self.load(model_dir)

    def _call_back(self):
        print("Disabling gradients !!")
        for params in self.depth_node.embed.parameters():
            params.requires_grad = False
