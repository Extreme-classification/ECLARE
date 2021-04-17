from xclib.utils.sparse import topk, retain_topk
import xclib.evaluation.xc_metrics as xc
from .model_base import ModelBase
import libs.features as feat
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import numpy as np
import torch
import time
import os


class ModelECLARE(ModelBase):
    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        super(ModelECLARE, self).__init__(params, net, criterion,
                                            optimizer, *args, **kwargs)

class ModelECLAREpp(ModelBase):
    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        super(ModelECLAREpp, self).__init__(params, net, criterion,
                                            optimizer, *args, **kwargs)
    
    def get_lbl_cent(self, dataset):
        encoder = self.net._get_encoder()
        dataset.mode = "test"
        docs = normalize(self._doc_embed(dataset, 0, encoder, True))
        dataset.mode = "train"
        y = dataset.labels.Y
        lbl_cnt = y.transpose().dot(docs)
        return lbl_cnt


class ModelECLAREfe(ModelBase):
    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        super(ModelECLAREfe, self).__init__(params, net, criterion,
                                            optimizer, *args, **kwargs)
        self.lbl_cnt = params.lbl_cnt
    
    def _prep_for_depth(self, depth, train_ds, valid_ds):
        torch.manual_seed(self.tree_idx)
        torch.cuda.manual_seed_all(self.tree_idx)
        np.random.seed(self.tree_idx)
        self.logger.info("learning for depth %d" % (depth))
        train_ds.mode = 'test'
        self._prep_ds_for_depth(depth, train_ds, valid_ds)
        self.net.cpu()
        self.net._prep_for_depth(depth)
        print(self.net)
        if depth == 0:
            document = self._doc_embed(train_ds, 0, self.net.depth_node, True)
            train_ds.feature_type = "dense"
            train_ds.features = feat.construct("", "", document, False, "dense")

            document = self._doc_embed(valid_ds, 0, self.net.depth_node, True)
            valid_ds.feature_type = "dense"
            valid_ds.features = feat.construct("", "", document, False, "dense")

        self.learning_rate = self.lrs[depth]
        self.dlr_step = self.dlr_steps[depth]
        self.optimizer.learning_rate = self.lrs[depth]
        self.optimizer.construct(self.net.depth_node, None)
        train_ds.mode = 'train'
        return self._prep_dl_for_depth(depth, train_ds, valid_ds)
    
    def get_lbl_cent(self, dataset):
        encoder = self.net._get_encoder()
        dataset.mode = "test"
        docs = normalize(self._doc_embed(dataset, 0, encoder, True))
        dataset.mode = "train"
        y = dataset.labels.Y
        lbl_cnt = normalize(y.transpose().dot(docs))
        return lbl_cnt
    
    def predict(self, data_dir, model_dir, dataset, data=None,
                ts_feat_fname='tst_X_Xf.txt', ts_label_fname='tst_X_Y.txt',
                batch_size=256, num_workers=6, keep_invalid=False,
                feature_indices=None, label_indices=None,
                normalize_features=True, normalize_labels=False, **kwargs):
        self.net.load(fname=model_dir)
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset), fname_features=ts_feat_fname,
            fname_labels=ts_label_fname, data=data,
            keep_invalid=keep_invalid, normalize_features=normalize_features,
            normalize_labels=normalize_labels, mode='test',
            feature_indices=feature_indices, label_indices=label_indices)

        encoder = self.net._get_encoder()
        docs = self._doc_embed(dataset, 0, encoder, True)
        dataset.feature_type = "dense"
        dataset.features = feat.construct("", "", docs, False, "dense")
        predicted_labels, _ = self._predict(dataset, model_dir, **kwargs)

        self._print_stats(dataset.labels.ground_truth, predicted_labels)
        return predicted_labels
