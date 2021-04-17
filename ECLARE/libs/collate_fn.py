import torch
import numpy as np
import scipy.sparse as sp
from xclib.utils.sparse import retain_topk
from xclib.evaluation import xc_metrics as xc


def _block_sparse_matrix(label_words):
    indx = torch.LongTensor(np.vstack([*label_words.nonzero()]))
    data = torch.FloatTensor(label_words.data)
    shape = torch.Size(label_words.shape)
    return indx, data, shape


def _to_sparse_matrix(cols, flags, num_lbls):
    num_inst = cols.shape[0]
    rows = np.arange(num_inst).reshape(-1, 1)
    s_flg = np.zeros((num_inst, num_lbls))
    s_mat = sp.lil_matrix((num_inst, num_lbls))
    s_flg = sp.lil_matrix((num_inst, num_lbls))
    s_flg[rows, cols] = flags
    data = np.ones(cols.shape).tolist()
    cols = cols.tolist()
    s_mat.rows[:] = cols
    s_mat.data[:] = data
    return s_mat, s_flg.tocsc()


def _paddedbatch(Var, padding_val,
                 dtype=torch.FloatTensor, *args):
    return torch.nn.utils.rnn.pad_sequence(
        list(map(lambda x: torch.from_numpy(x).type(dtype), Var)),
        batch_first=True, padding_value=padding_val)


class construct_collate_fn:
    def __init__(self, feature_type, use_shortlist=False, freeze_params=False,
                 num_partitions=1, sparse_label_fts=None, traverse=True,
                 sparse_graph=None, padding_idx=0, num_labels=-1, mode="test"):
        self.num_partitions = num_partitions
        self.padding_idx = padding_idx
        self.sparse_label_fts = sparse_label_fts
        self.sparse_graph = sparse_graph
        self.traverse = traverse
        self.freeze_params = freeze_params
        if self.sparse_label_fts is not None:
            self.sparse_label_fts = self.sparse_label_fts.tocsr()
        if self.sparse_graph is not None:
            self.sparse_graph = sparse_graph.tocsr()
        self.n_lbs = num_labels
        self.use_shortlist = use_shortlist
        self.collate_docs = self._get_docs(feature_type)
        self.mode = mode
        self.batcher = self._batcher(sparse_label_fts, use_shortlist)

    def setup(self, dataset):
        pass

    def __call__(self, batch):
        return self.batcher(batch)

    def _batcher(self, sparse_label_fts, use_shortlist):
        if sparse_label_fts is not None:
            if use_shortlist:
                if self.traverse:
                    print("traversing the graph")
                    return self.collate_fn_shorty_lbf_rl
                else:
                    print("Not traversing the graph")
                    return self.collate_fn_shorty_lbf
            return self.collate_fn_full
        else:
            if use_shortlist:
                return self.collate_fn_shorty
            return self.collate_fn_full

    def _get_docs(self, feature_type):
        if feature_type == 'dense':
            return self.collate_fn_docs_dense
        elif feature_type == 'sparse':
            return self.collate_fn_docs_sparse
        else:
            print("Kuch bhi")

    def collate_fn_docs_dense(self, batch, b_data):
        b_data['X'] = torch.stack(list(
            map(lambda x: torch.from_numpy(x[0]), batch)
        ), 0).type(torch.FloatTensor)
        b_data['batch_size'] = len(batch)
        b_data['idx'] = np.arange(b_data['batch_size']).reshape(-1, 1)
        b_data['is_sparse'] = False

    def collate_fn_docs_sparse(self, batch, b_data):
        s_docs = sp.vstack(list(map(lambda x: x[0], batch))).tocsr()
        b_data['X_ptr'] = torch.from_numpy(
            s_docs.indptr[:-1]).type(torch.LongTensor)
        b_data['X_ind'] = torch.from_numpy(
            s_docs.indices).type(torch.LongTensor)
        b_data['X_wts'] = torch.from_numpy(s_docs.data).type(torch.FloatTensor)
        b_data['batch_size'] = len(batch)
        b_data['idx'] = np.arange(len(batch)).reshape(-1, 1)
        b_data['is_sparse'] = True

    def collate_fn_shorty_lbf_rl(self, batch):
        b_data = {}
        self.collate_docs(batch, b_data)
        Y_s = sp.vstack(list(map(lambda x: x[2], batch))).tocsr()
        # print(Y_s.shape, self.sparse_graph.shape)
        s_nbr = Y_s.dot(self.sparse_graph)
        lbl_pred = np.where(s_nbr.getnnz(axis=0) > 0)[0]

        f_shoty = torch.from_numpy(lbl_pred)
        b_data['f_shoty'] = f_shoty

        if not self.freeze_params:
            shoty_lg = self.sparse_graph[lbl_pred].tocsc()
            shoty_lg = shoty_lg[:, lbl_pred].tocsr()
            shoty_lf = self.sparse_label_fts[lbl_pred]
            lf = np.where(shoty_lf.getnnz(axis=0) > 0)[0]
            shoty_lf = shoty_lf.tocsc()[:, lf].tocsr()
            b_data['shoty_lg'] = _block_sparse_matrix(shoty_lg)
            b_data['lf'] = torch.from_numpy(lf).type(torch.LongTensor)
            b_data['shoty_lf'] = _block_sparse_matrix(shoty_lf)

        Y_r = torch.zeros(self.n_lbs+1, dtype=torch.long)
        Y_r[f_shoty] = torch.arange(lbl_pred.size, dtype=torch.long)
        indptr, indices = s_nbr.indptr, s_nbr.indices
        _Y_s = _paddedbatch(map(lambda x: indices[x[0]:x[1]],
                                zip(indptr[:-1], indptr[1:])),
                                self.n_lbs, dtype=torch.LongTensor)
        b_data['Y_r'] = Y_r[_Y_s]
        if self.mode == "train":
            Y = sp.vstack(list(map(lambda x: x[1], batch))).tocsc()
            # print(Y.shape)
            Y = np.take_along_axis(Y, _Y_s.numpy(), axis=1).todense()
            b_data['Y'] = torch.from_numpy(Y).type(torch.FloatTensor)
        return b_data

    def collate_fn_shorty_lbf(self, batch):
        b_data = {}
        self.collate_docs(batch, b_data)

        Y_s = sp.vstack(list(map(lambda x: x[2], batch))).tocsr()
        indptr, indices = Y_s.indptr, Y_s.indices
        Y_s = _paddedbatch(map(lambda x: indices[x[0]:x[1]],
                               zip(indptr[:-1], indptr[1:])),
                               self.n_lbs, dtype=torch.LongTensor)

        f_shoty = torch.unique(Y_s)
        b_data['f_shoty'] = f_shoty
        lbl_pred = f_shoty.numpy()

        if not self.freeze_params:
            shoty_lg = self.sparse_graph[lbl_pred].tocsc()
            shoty_lg = shoty_lg[:, lbl_pred].tocsr()
            shoty_lf = self.sparse_label_fts[lbl_pred]
            lf = np.where(shoty_lf.getnnz(axis=0) > 0)[0]
            shoty_lf = shoty_lf.tocsc()[:, lf].tocsr()
            b_data['lf'] = torch.from_numpy(lf).type(torch.LongTensor)
            b_data['shoty_lg'] = _block_sparse_matrix(shoty_lg)
            b_data['shoty_lf'] = _block_sparse_matrix(shoty_lf)

        Y_r = torch.zeros(self.n_lbs+1, dtype=torch.long)
        Y_r[f_shoty] = torch.arange(lbl_pred.size, dtype=torch.long)
        b_data['Y_r'] = Y_r[Y_s]

        if self.mode == "train":
            Y = sp.vstack(list(map(lambda x: x[1], batch))).tocsc()
            # print(Y.shape)
            Y = np.take_along_axis(Y, Y_s.numpy(), axis=1).todense()
            b_data['Y'] = torch.from_numpy(Y).type(torch.FloatTensor)
        return b_data

    def collate_fn_shorty(self, batch):
        """
            Combine each sample in a batch with shortlist
            For sparse features
        """
        b_data = {}
        self.collate_docs(batch, b_data)

        Y_s = sp.vstack(list(map(lambda x: x[2], batch))).tocsr()
        indptr, indices, data = Y_s.indptr, Y_s.indices, Y_s.data

        b_data['Y_s'] = _paddedbatch(map(lambda x: indices[x[0]:x[1]],
                                         zip(indptr[:-1], indptr[1:])),
                                         self.n_lbs, dtype=torch.LongTensor)

        if self.mode == "train":
            Y = sp.vstack(list(map(lambda x: x[0][1], batch))).tocsc()
            Y = np.take_along_axis(Y, b_data['Y_s'].numpy(), axis=1).todense()
            b_data['Y'] = torch.from_numpy(Y).type(torch.FloatTensor)

        b_data['Y_d'] = _paddedbatch(map(lambda x: data[x[0]:x[1]],
                                         zip(indptr[:-1], indptr[1:])),
                                         -np.inf, dtype=torch.FloatTensor)
        return b_data

    def collate_fn_full(self, batch):
        """
            Combine each sample in a batch
            For dense features
        """
        b_data = {}
        self.collate_docs(batch, b_data)
        if self.mode == "train":
            b_data['Y'] = torch.from_numpy(sp.vstack(
                list(map(lambda x: x[1], batch))).todense()).type(torch.FloatTensor)
        return b_data
