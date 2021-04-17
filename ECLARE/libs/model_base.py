from scipy.sparse import lil_matrix, issparse
from .collate_fn import construct_collate_fn
from xclib.utils.sparse import retain_topk
import xclib.evaluation.xc_metrics as xc
from .dataset import construct_dataset
from torch.utils.data import DataLoader
from libs.utils import load_overlap
from .tracking import Tracking
import libs.features as feat
import scipy.sparse as sp
import numpy as np
import logging
import torch
import time
import sys
import os


class ModelBase:
    def __init__(self, params, net, criterion, optimizer, *args, **kwargs):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = 0.001
        self.current_epoch = 0
        self.last_saved_epoch = -1
        self.model_dir = params.model_dir
        self.label_padding_index = params.label_padding_index
        self.last_epoch = 0
        self.feature_type = params.feature_type
        self.shortlist_size = -1
        self.dlr_step = 0
        self.dlr_factor = params.dlr_factor
        self.progress_step = 500
        self.verbose_lbs = params.verbose
        self.freeze_embeddings = params.freeze_embeddings
        self.model_fname = params.model_fname
        self.tree_idx = params.tree_idx + 1
        self.logger = self.get_logger(name=self.model_fname)
        self.embedding_dims = params.embedding_dims
        self.filter_docs, self.filter_lbls = load_overlap(
            os.path.join(params.data_dir, params.dataset),
            params.label_indices, filter_label_file=params.filter_labels)
        self.method = params.cluster_method
        self.lrs = params.depth_lrs
        self.call_back = params.call_backs
        self.beam = params.beam_size
        self.batch_size = params.batch_sizes
        self.dlr_steps = params.dlr_steps
        self.model_dir = params.model_dir
        self.traverse = params.traverse
        self.tracking = Tracking()

    def transfer_to_devices(self):
        self.net.to()

    def _prep_ds(self, depth, _dataset, mode='train'):
        _pred, trav = None, self.traverse
        if depth > 0:
            _dataloader = self._prep_dl(
                depth-1, _dataset, mode='test', traverse=trav)
            _pred, _ = self._predict_depth(depth-1, _dataloader, trav)
            self._print_stats(_dataset.labels.ground_truth, _pred)
        clusters = self.net._clusters(depth)
        _dataset._prep_for_depth(np.bool(depth > 0), clusters)
        _dataset.build_shortlist(_pred, self.net)

    def _traverse_graph(self, shorty, k, batch=512):
        self.logger.info("Traversing graph: %d" % k)
        num_inst, _ = shorty.shape
        shorty.data[:] = np.exp(shorty.data)
        batches_inst = np.ceil(num_inst/batch)
        mats, batch_pred_time = [], 0
        for idx, i in enumerate(range(0, num_inst, batch)):
            if idx % 500 == 0:
                self.logger.info("Traversing [%d/%d]" % (idx, batches_inst))
            start, end = i, min(i+batch, num_inst)
            begin = time.time()
            sub_mat = self.net.traverse_graph(shorty[start: end])
            batch_pred_time += time.time() - begin
            mats.append(retain_topk(sub_mat, k=k))
        t_shorty = sp.vstack(mats).tocsr()
        print("After traversing", np.min(t_shorty.data), np.max(t_shorty.data))
        shorty = shorty.multiply(0.5) + t_shorty.multiply(0.5)
        shorty = shorty.tocsr()
        print("After traversing", np.min(shorty.data), np.max(shorty.data))
        shorty.data[:] = np.log(shorty.data)
        return shorty, batch_pred_time

    def _prep_dl(self, d, _dataset, mode, traverse=False):
        if mode == "train":
            return self._create_dl(_dataset, batch_size=self.batch_size[d],
                                   num_workers=6, shuffle=True,
                                   mode=mode, use_shortlist=np.bool(d > 0),
                                   sparse_label_fts=self.net.label_words,
                                   sparse_graph_fts=self.net.graph)

        else:
            return self._create_dl(_dataset, batch_size=self.batch_size[d],
                                   num_workers=6, shuffle=False, traverse=False,
                                   mode=mode, use_shortlist=np.bool(d > 0))

    def _prep_ds_for_depth(self, depth, train_dataset, valid_dataset):
        self._prep_ds(depth, valid_dataset, mode='test')
        self._prep_ds(depth, train_dataset, mode='train')

    def _prep_dl_for_depth(self, depth, train_dataset, valid_dataset):
        valid_dl = self._prep_dl(depth, valid_dataset, mode='test')
        train_dl = self._prep_dl(depth, train_dataset, mode='train')
        return train_dl, valid_dl

    def _prep_for_depth(self, depth, train_ds, valid_ds):
        torch.manual_seed(self.tree_idx)
        torch.cuda.manual_seed_all(self.tree_idx)
        np.random.seed(self.tree_idx)
        self.logger.info("learning for depth %d" % (depth))
        self._prep_ds_for_depth(depth, train_ds, valid_ds)
        self.net.cpu()
        self.net._prep_for_depth(depth)
        print(self.net)
        self.learning_rate = self.lrs[depth]
        self.dlr_step = self.dlr_steps[depth]
        self.optimizer.learning_rate = self.learning_rate
        self.optimizer.construct(self.net.depth_node, None)
        return self._prep_dl_for_depth(depth, train_ds, valid_ds)

    def _create_dataset(self, data_dir, fname_features, fname_labels=None,
                        data=None, mode='predict', normalize_features=True,
                        normalize_labels=False, feature_type=None,
                        keep_invalid=False, feature_indices=None,
                        label_indices=None, size_shortlist=None,
                        shortlist_method='static', shorty=None,
                        classifier_type="GraphXMLTree"):
        size_shortlist = self.shortlist_size \
            if size_shortlist is None else size_shortlist
        feature_type = self.feature_type \
            if feature_type is None else feature_type
        _dataset = construct_dataset(
            data_dir=data_dir, fname_features=fname_features,
            fname_labels=fname_labels, data=data,
            model_dir=self.model_dir, size_shortlist=size_shortlist,
            mode=mode, normalize_features=normalize_features,
            normalize_labels=normalize_labels, keep_invalid=keep_invalid,
            num_centroids=1, feature_type=feature_type,
            num_clf_partitions=1, feature_indices=feature_indices,
            label_indices=label_indices, shortlist_method=shortlist_method,
            shorty=shorty, classifier_type=classifier_type)
        return _dataset

    def _create_dl(self, dataset, batch_size=128, num_workers=4,
                   shuffle=False, use_shortlist=False, mode='test',
                   sparse_label_fts=None, sparse_graph_fts=None,
                   traverse=True):
        """
            Create data loader for given dataset
        """
        collate_fn = construct_collate_fn(
            dataset.feature_type, use_shortlist,
            sparse_label_fts=sparse_label_fts,
            sparse_graph=sparse_graph_fts,
            padding_idx=dataset.features.num_features,
            num_labels=dataset.num_labels, traverse=traverse,
            freeze_params=self.freeze_embeddings, mode=mode)
        if mode == 'train':
            collate_fn.setup(dataset)
        dt_loader = DataLoader(dataset, batch_size=batch_size,
                               num_workers=num_workers, shuffle=shuffle,
                               collate_fn=collate_fn)
        return dt_loader

    def get_logger(self, name='CoXML', level=logging.INFO):
        """
            Return logging object!
        """
        logging.basicConfig(level=level, stream=sys.stdout)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        return logger

    def _compute_loss(self, out_ans, batch_data):
        device = out_ans.get_device()
        _true = batch_data['Y'].to(device)
        return self.criterion(out_ans, _true).to(device)

    def _step(self, data_loader, batch_div=False):
        """
            Training step
        """
        self.net.to()
        self.net.train()
        torch.set_grad_enabled(True)
        num_batches, mean_loss = len(data_loader), 0
        for batch_idx, batch_data in enumerate(data_loader):
            batch_size = batch_data['batch_size']
            out_ans = self.net.forward(batch_data)
            loss = self._compute_loss(out_ans, batch_data)
            mean_loss += loss.item()*batch_size
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch_idx % self.progress_step == 0:
                self.logger.info(
                    "Training progress: [{}/{}]".format(
                        batch_idx, num_batches))
            del batch_data
        self.net.cpu()
        return mean_loss / data_loader.dataset.num_instances

    def _print_stats(self, Y, mat):
        _prec, _ndcg, _recall = self.evaluate(Y, mat)
        _max_k = _recall.size
        self.tracking.val_precision.append(_prec)
        self.tracking.val_precision.append(_ndcg)
        self.logger.info("R@{}: {}, R@{}: {}, R@{}: {}, R@{}: {}".format(
            1, _recall[0]*100, 3, _recall[2]*100,
            self.beam, _recall[self.beam-1]*100,
            _max_k, _recall[_max_k-1]*100))
        self.logger.info("P@{}: {}, P@{}: {}, P@{}: {}, P@{}: {}".format(
            1, _prec[0]*100, 3, _prec[2]*100,
            self.beam, _prec[self.beam-1]*100,
            _max_k, _prec[_max_k-1]*100))

    def validate(self, valid_loader, model_dir=None, epoch=None):
        predicted_labels, prediction_time = self._predict_depth(
            self.net.depth, valid_loader)
        self.tracking.validation_time = self.tracking.validation_time \
            + prediction_time
        Y = valid_loader.dataset.labels.ground_truth
        self._print_stats(Y, predicted_labels)

    def _predict_depth(self, depth, data_loader, traverse=True, **kwargs):
        self.net.cpu()
        self.net.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        self.net.to()
        num_inst = data_loader.dataset.num_instances
        num_lbls = data_loader.dataset.num_labels
        num_batches = len(data_loader)
        pred_lbs = lil_matrix((num_inst, num_lbls + self.net.offset))
        batch_pred_time, count = 0, 0
        for batch_idx, batch_data in enumerate(data_loader):
            time_begin = time.time()
            score = self.net._predict(batch_data)
            if depth > 0:
                index = batch_data['Y_s'].cuda()
                _val = batch_data['Y_d'].cuda()
                score = torch.add(_val, score)
            else:
                score, index = torch.topk(score, self.beam)
            batch_pred_time += time.time()-time_begin
            score, index = score.cpu().numpy(), index.cpu().numpy()
            self._update_in_sparse(count, index, score, pred_lbs)
            if batch_idx % self.progress_step == 0:
                self.logger.info("Prediction progress: [{}/{}]".format(
                    batch_idx, num_batches))
            count += index.shape[0]
        pred_lbs = retain_topk(pred_lbs.tocsr(), copy=False, k=self.beam)
        if traverse:
            self.logger.info("Traversing graph for predictions")
            pred_lbs, _time = self._traverse_graph(pred_lbs, self.beam,
                                                   self.batch_size[depth])
            batch_pred_time += _time
        avg_time = batch_pred_time*1000/pred_lbs.shape[0]
        self.logger.info("Avg pred time {:0.4f} ms".format(avg_time))
        if self.net.offset:
            pred_lbs = pred_lbs.tocsc()[:, :-1].tocsr()
        self.net.cpu()
        return pred_lbs.tocsr(), batch_pred_time

    def _train_depth(self, train_ds, valid_ds, model_dir,
                     depth, validate_after=5):

        train_dl, valid_dl = self._prep_for_depth(depth, train_ds, valid_ds)
        fname = os.path.join(model_dir, self.model_fname +
                             "-depth=%d_params.pkl" % (depth))
        if os.path.exists(fname):
            self.logger.info("Using pre-trained at "+str(depth))
            self.load(model_dir, depth)
            return
        num_epochs = self.call_back[depth]
        counter_next_decay = self.dlr_step
        for epoch in range(0, num_epochs):
            if counter_next_decay == 0:
                self._adjust_parameters()
                counter_next_decay = self.dlr_step
            start_time = time.time()
            tr_avg_loss = self._step(train_dl)
            end_time = time.time()
            self.tracking.mean_train_loss.append(tr_avg_loss)
            self.tracking.train_time += end_time - start_time
            self.logger.info("Epoch: {}, loss: {}, time: {} sec".format(
                epoch, tr_avg_loss, end_time - start_time))
            if valid_dl is not None and epoch % validate_after == 0:
                self.validate(valid_dl, model_dir, epoch)
                self.save(model_dir, depth)
            counter_next_decay -= 1
            self.tracking.last_epoch += 1
        if valid_dl is not None and not (epoch % validate_after == 0):
            self.validate(valid_dl, model_dir, epoch)
        self.save(model_dir, depth)

    def _fit(self, train_ds, valid_ds, model_dir, result_dir, validate_after=5):
        for depth in np.arange(len(self.batch_size)):
            self._train_depth(train_ds, valid_ds, model_dir,
                              depth, validate_after)
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()

        self.tracking.save(os.path.join(result_dir, 'training_statistics.pkl'))
        self.logger.info(
            "Training time: {} sec, Validation time: {} sec"
            ", Shortlist time: {} sec, Model size: {} MB".format(
                self.tracking.train_time, self.tracking.validation_time,
                self.tracking.shortlist_time, self.net.model_size))

    def get_lbl_cent(self, dataset):
        _labels = dataset.labels.Y.transpose()
        _features = dataset.features.X
        lbl_cnt = _labels.dot(_features).tocsr()
        lbl_cnt = retain_topk(lbl_cnt.tocsr(), k=1000).tocsr()
        return lbl_cnt

    def build_net(self, dataset):
        labels_repr = self.get_lbl_cent(dataset)
        freq_y = np.ravel(dataset.labels.Y.sum(axis=0))
        verb_lbs, norm_lbs = np.asarray([]), np.arange(freq_y.size)
        if self.verbose_lbs > 0:
            verb_lbs = np.where(freq_y > self.verbose_lbs)[0]
            norm_lbs = np.where(freq_y <= self.verbose_lbs)[0]
        print("Label centroids shape:", labels_repr.shape)
        build_start = time.time()
        self.net.build(labels_repr, norm_lbs, verb_lbs,
                       model_dir=self.model_dir)
        build_end = time.time()
        build_time = build_end-build_start
        self.tracking.train_time += build_time

    def fit(self, data_dir, model_dir, result_dir, dataset,
            data=None, tr_feat_fname='trn_X_Xf.txt',
            tr_label_fname='trn_X_Y.txt', val_feat_fname='tst_X_Xf.txt',
            val_label_fname='tst_X_Y.txt', num_workers=4,
            init_epoch=0, keep_invalid=False,
            feature_indices=None, label_indices=None, normalize_features=True,
            normalize_labels=False, validate=False,
            validate_after=5, **kwargs):
        self.logger.info("Loading training data.")

        train_dataset = self._create_dataset(
            os.path.join(data_dir, dataset),
            fname_features=tr_feat_fname,
            fname_labels=tr_label_fname,
            data=data, mode='train',
            keep_invalid=keep_invalid,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            feature_indices=feature_indices,
            label_indices=label_indices)
        if validate:
            self.logger.info("Loading validation data.")
            valid = self._create_dataset(
                os.path.join(data_dir, dataset),
                fname_features=val_feat_fname,
                fname_labels=val_label_fname,
                data={'X': None, 'Y': None}, mode='test',
                keep_invalid=keep_invalid,
                normalize_features=normalize_features,
                normalize_labels=normalize_labels,
                feature_indices=feature_indices,
                label_indices=label_indices)
            valid.labels.Y = self._filter_labels(valid.labels.Y)
        self.build_net(train_dataset)
        self._fit(train_dataset, valid, model_dir, result_dir, validate_after)

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
        predicted_labels, _ = self._predict(dataset, model_dir, **kwargs)
        self._print_stats(dataset.labels.ground_truth, predicted_labels)
        return predicted_labels

    def _predict(self, dataset, model_dir, **kwargs):
        self.logger.info("Preping the classifile")
        for depth in range(0, self.net.height):
            self.logger.info("Predicting for depth {}".format(depth))
            self._prep_ds(depth, dataset, mode='test')
            self.net._prep_for_depth(depth, initialize=False)
            self.load(model_dir, depth)
        data_loader = self._prep_dl(depth, dataset, mode='test')
        return self._predict_depth(depth, data_loader, self.traverse)

    def extract(self, model_dir):
        self.net.load(fname=model_dir)
        self.net._prep_for_depth(0, initialize=False)
        self.load(model_dir, 0)
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        return self.net.depth_node.embed.cpu().state_dict()

    def get_document(self, data_dir, model_dir, dataset, data=None,
                     ts_feat_fname='tst_X_Xf.txt', if_labels=False,
                     batch_size=256, num_workers=6, keep_invalid=False,
                     feature_indices=None, label_indices=None, 
                     normalize_features=True, normalize_labels=False, **kwargs):
        self.net.load(fname=model_dir)
        self.net._prep_for_depth(0, initialize=False)
        self.load(model_dir, 0)
        dataset = self._create_dataset(
            os.path.join(data_dir, dataset), fname_features=ts_feat_fname,
            fname_labels=ts_feat_fname, data=data,
            keep_invalid=keep_invalid, normalize_features=normalize_features,
            normalize_labels=normalize_labels, mode='test',
            feature_indices=feature_indices, label_indices=label_indices)
        self.net.cpu()
        print(self.net)
        self.net.to()
        docs = self._doc_embed(dataset, 0, self.net.depth_node, if_labels)
        return docs

    def _doc_embed(self, dataset, depth, encoder, return_coarse=False, **kwargs):
        encoder.to()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        encoder.eval()
        data_loader = self._prep_dl(depth, dataset, mode='test')
        num_batches = len(data_loader)
        docs = []
        for batch_idx, batch_data in enumerate(data_loader):
            docs.append(encoder.encode(
                batch_data, return_coarse).cpu().numpy())
            if batch_idx % self.progress_step == 0:
                self.logger.info("Fectching progress: [{}/{}]".format(
                    batch_idx, num_batches))
        encoder.cpu()
        docs = np.vstack(docs)
        return docs

    def _update_in_sparse(self, counter, predicted_label_index, predicted_label_scores, sparse_mat):
        index = np.arange(
            predicted_label_index.shape[0]).reshape(-1, 1) + counter
        sparse_mat[index, predicted_label_index] = predicted_label_scores

    def _adjust_parameters(self):
        self.optimizer.adjust_lr(self.dlr_factor)
        self.learning_rate *= self.dlr_factor
        self.dlr_step = max(5, self.dlr_step//2)
        self.logger.info(
            "Adjusted learning rate to: {}".format(self.learning_rate))

    def save(self, model_dir, depth, *args):
        fname = self.model_fname + "-depth=%d" % (depth)
        file_path = os.path.join(model_dir, fname+"_params.pkl")
        torch.save(self.net.state_dict(), file_path)

    def load(self, model_dir, depth,
             label_padding_index=None, *args):
        fname = self.model_fname + "-depth=%d" % (depth)
        file_path = os.path.join(model_dir, fname+"_params.pkl")
        self.net.load_state_dict(torch.load(file_path))
        self.transfer_to_devices()

    def _evaluate(self, true_labels, predicted_labels):
        pmat = predicted_labels.tocsr()
        acc = xc.Metrics(true_labels)
        rec = xc.recall(pmat, true_labels, self.beam)
        _p, _n = acc.eval(predicted_labels.tocsr(), self.beam)
        return _p, _n, rec

    def evaluate(self, true_labels, predicted_labels, _filter=True):
        if self.net.depth == self.net.height - 1 and _filter:
            true_labels = self._filter_labels(true_labels)
            _predicted_labels = self._filter_labels(predicted_labels)
        else:
            _predicted_labels = predicted_labels
        return self._evaluate(true_labels, _predicted_labels)

    def _filter_labels(self, score_mat):
        if len(self.filter_docs) > 0:
            self.logger.info("Filtering labels to remove overlap")
            return_score_mat = score_mat.copy().tolil()
            return_score_mat[self.filter_docs, self.filter_lbls] = 0
            return_score_mat = return_score_mat.tocsr()
            return return_score_mat
        else:
            return score_mat
