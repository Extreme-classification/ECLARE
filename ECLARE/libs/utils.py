import numpy as np
import scipy.sparse as sp
from xclib.utils import graph
from xclib.utils import sparse as xs
import json
import os
import re

def normalize_graph(mat, axis=1):
    diags = mat.diagonal()
    print("Zero Diagonals:", np.sum(diags == 0))
    col_nnz = np.sqrt(1/np.ravel(mat.sum(axis=0)))
    row_nnz = np.sqrt(1/np.ravel(mat.sum(axis=1)))
    c_diags = sp.diags(col_nnz)
    r_diags = sp.diags(row_nnz)
    mat = r_diags.dot(mat).dot(c_diags)
    mat.eliminate_zeros()
    return mat


def load_overlap(data_dir, valid_labels, filter_label_file='filter_labels'):
    docs = np.asarray([])
    lbs = np.asarray([])
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        print("Loading from pre-build file")
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        if filter_lbs.size > 0:
            docs = filter_lbs[:, 0]
            lbs = filter_lbs[:, 1]
            valid_labels = np.loadtxt(valid_labels, dtype=np.int32)
            max_lbl = max(max(lbs), max(valid_labels))+1
            hash_key = np.ones(max_lbl)*-1
            hash_key[valid_labels] = np.arange(valid_labels.size)
            lbs_remapped = hash_key[lbs]
            valid_idx = np.where(lbs_remapped != -1)[0]
            docs = docs[valid_idx]
            lbs = lbs_remapped[valid_idx]
    print("Overlap is:", docs.size)
    return docs, lbs


def append_padding_embedding(embeddings):
    embedding_dim = embeddings.shape[1]
    app = np.zeros((1, embedding_dim))
    return np.vstack([embeddings, app])


def save_predictions(preds, result_dir, prefix='predictions'):
    sp.save_npz(os.path.join(result_dir, '{}.npz'.format(prefix)), preds)


def resolve_schema_args(jfile, ARGS):
    """
        Reads JSON and complete the parameters from ARGS
    """
    arguments = re.findall(r"#ARGS\.(.+?);", jfile)
    for arg in arguments:
        replace = '#ARGS.%s;' % (arg)
        to = str(ARGS.__dict__[arg])
        if jfile.find('\"#ARGS.%s;\"' % (arg)) != -1:
            replace = '\"#ARGS.%s;\"' % (arg)
            if isinstance(ARGS.__dict__[arg], str):
                to = str("\""+ARGS.__dict__[arg]+"\"")
        jfile = jfile.replace(replace, to)
    return jfile


def fetch_json(file, ARGS):
    with open(file, encoding='utf-8') as f:
        file = ''.join(f.readlines())
        schema = resolve_schema_args(file, ARGS)
    return json.loads(schema)


class PrunedWalk(graph.RandomWalk):
    def __init__(self, Y, valid_labels=None, yf=None):
        super(PrunedWalk, self).__init__(Y, valid_labels)
        self.yf = yf
        if self.yf is not None:
            self.yf = yf[self.valid_labels]
            print(self.yf.shape)

    def simulate(self, walk_to=100, p_reset=0.2, k=None, b_size=1000, max_dist=2):
        q_lbl = self.Y.indices
        q_rng = self.Y.indptr
        trn_y = self.Y.transpose().tocsr()
        trn_y.sort_indices()
        trn_y.eliminate_zeros()
        l_qry = trn_y.indices
        l_rng = trn_y.indptr
        n_lbs = self.Y.shape[1]
        zeros = 0
        mats = []
        pruned_edges = 0
        for p_idx, idx in enumerate(np.arange(0, n_lbs, b_size)):
            if p_idx % 50 == 0:
                print("INFO:WALK: completed [ %d/%d ]" % (idx, n_lbs))
            start, end = idx, min(idx+b_size, n_lbs)
            cols, data = graph._random_walk(q_rng, q_lbl, l_rng, l_qry, walk_to,
                                      p_reset, start=start, end=end)
            rows = np.arange(end-start).reshape(-1, 1)
            rows = np.repeat(rows, walk_to, axis=1).flatten()
            mat = sp.coo_matrix((data, (rows, cols)), dtype=np.float32,
                                shape=(end-start, n_lbs))
            mat.sum_duplicates()
            mat = mat.tocsr()
            mat.sort_indices()
            if self.yf is not None:
                _rows, _cols = mat.nonzero()
                _lbf = self.yf[start+_rows]
                _dist = 1-np.ravel(np.sum(_lbf*self.yf[_cols], axis=1))
                mat.data[_dist > max_dist] = 0
                pruned_edges += np.sum(_dist > max_dist)
                mat.eliminate_zeros()
            diag = mat.diagonal(k=start)
            if k is not None:
                mat = xs.retain_topk(mat, k=k)
            _diag = mat.diagonal(k=start)
            _diag[_diag == 0] = diag[_diag == 0]
            zeros += np.sum(_diag == 0)
            _diag[_diag == 0] = 1
            mat.setdiag(_diag, k=start)
            mats.append(mat)
            del rows, cols
        print("INFO:WALK: completed [ %d/%d ]" % (n_lbs, n_lbs))
        mats = sp.vstack(mats).tocsr()
        rows, cols = mats.nonzero()
        r_mat = sp.coo_matrix((mats.data, (rows, cols)), dtype=np.float32,
                              shape=(self.num_lbls, self.num_lbls))
        r_mat = xs._map(r_mat, self.valid_labels, axis=0, shape=r_mat.shape)
        r_mat = xs._map(r_mat, self.valid_labels, axis=1, shape=r_mat.shape)
        return r_mat.tocsr()


def print_stats(mat, k=10):
    _mat = mat.copy()
    _mat.__dict__['data'][:] = 1
    freqs = _mat.sum(axis=1)
    print(np.max(freqs), np.min(freqs), np.std(freqs))
