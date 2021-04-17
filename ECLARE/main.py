"""
Modified from DECAF:
@InProceedings{Mittal21,
    author = "Mittal, A. and Dahiya, K. and Agrawal, S. and Saini, D. and Agarwal, S. and Kar, P. and Varma, M.",
    title = "DECAF: Deep Extreme Classification with Label Features",
    booktitle = "Proceedings of the ACM International Conference on Web Search and Data Mining",
    month = "March",
    year = "2021",
    }
"""
import os
import sys
import torch
import numpy as np
import libs.model as mw
import _pickle as pickle
import libs.utils as utils
import libs.parameters as parameters
import models.extreme_classifier as dp
import libs.optimizer_utils as optimizer_utils

__author__ = 'AM'


def load_embeddings(params):
    """Load word embeddings from numpy file
    * Support for:
        - loading pre-trained embeddings
        - loading head embeddings
    * vocabulary_dims must match #rows in embeddings
    """
    fname = params.embeddings
    print(fname)
    if os.path.exists(fname):
        print("Loading embeddings from file: {}".format(fname))
        if fname.endswith("pkl"):
            embeddings = torch.load(fname)
        if fname.endswith('npy'):
            embeddings = np.load(fname)
            indices = np.loadtxt(params.feature_indices, dtype=np.int)
            embeddings = embeddings[indices]
            embeddings = utils.append_padding_embedding(embeddings)
            weight = torch.from_numpy(embeddings)
            embeddings = {'weight': torch.from_numpy(embeddings)}
    else:
        print("Using random init")
        indices = np.loadtxt(params.feature_indices, dtype=np.int)
        embeddings = np.zeros((indices.size+1,
                                params.embedding_dims),
                                dtype=np.float32)
        weight = torch.from_numpy(embeddings)
        torch.nn.init.kaiming_uniform_(weight)
        embeddings = {'weight': weight}
    return embeddings


def train(model, params):
    model.fit(
        data_dir=params.data_dir, model_dir=params.model_dir,
        result_dir=params.result_dir, dataset=params.dataset,
        data={'X': None, 'Y': None},
        tr_feat_fname=params.tr_feat_fname,
        val_feat_fname=params.val_feat_fname,
        tr_label_fname=params.tr_label_fname,
        val_label_fname=params.val_label_fname,
        num_workers=params.num_workers, normalize_features=params.normalize,
        validate=params.validate,
        keep_invalid=params.keep_invalid,
        shortlist_method=params.cluster_method,
        validate_after=params.validate_after,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices)


def inference(model, params):
    predicted_labels = model.predict(
        data_dir=params.data_dir,
        dataset=params.dataset,
        model_dir=params.model_dir,
        ts_label_fname=params.ts_label_fname,
        ts_feat_fname=params.ts_feat_fname,
        normalize_features=params.normalize,
        num_workers=params.num_workers,
        top_k=params.top_k,
        data={'X': None, 'Y': None},
        keep_invalid=params.keep_invalid,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices,
        shortlist_method=params.cluster_method)
    # Real number of labels
    utils.save_predictions(predicted_labels, params.result_dir,
                           prefix=params.pred_fname)


def extract(model, params):
    embeddings = model.extract(params.model_dir)
    emb_p = os.path.join(params.model_dir, "embeddings.pkl")
    torch.save(embeddings, emb_p)


def get_documents(model, params):
    """Generates document embeddings
    Parameters
    ----------
    model: ECLARE
        train this model (typically ECLARE model)
    params: NameSpace
        parameter of the model
    """
    documents = model.get_document(
        data_dir=params.data_dir,
        dataset=params.dataset,
        model_dir=params.model_dir,
        ts_feat_fname=params.ts_feat_fname,
        normalize_features=params.normalize,
        num_workers=params.num_workers,
        data={'X': None, 'Y': None},
        keep_invalid=params.keep_invalid,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices,
        shortlist_method=params.cluster_method,
        if_labels=False,
    )
    np.save(os.path.join(params.result_dir,
                         params.pred_fname),
            documents)


def get_labels(model, params):
    """Generates label embeddings
    Parameters
    ----------
    model: ECLARE
        train this model (typically ECLARE model)
    params: NameSpace
        parameter of the model
    """
    documents = model.get_document(
        data_dir=params.data_dir,
        dataset=params.dataset,
        model_dir=params.model_dir,
        ts_feat_fname=params.ts_feat_fname,
        normalize_features=params.normalize,
        num_workers=params.num_workers,
        data={'X': None, 'Y': None},
        keep_invalid=params.keep_invalid,
        feature_indices=params.feature_indices,
        label_indices=params.label_indices,
        shortlist_method=params.cluster_method,
        if_labels=True,
    )
    np.save(os.path.join(params.result_dir,
                         params.pred_fname),
            documents)

def construct_network(params):
    if params.model_fname in ['ECLARE++', 'ECLARE-S']:
        return dp.ECLAREpp(params)
    elif params.model_fname in ['ECLARE', 'ECLARE-v']:
        return dp.ECLAREfe(params)
    else:
        raise NotImplementedError("Not tested yet!")


def construct_model(params, net, criterion, optimizer, shorty):
    if params.model_fname in ['ECLARE', 'ECLARE-v']:
        return mw.ModelECLAREfe(params, net, criterion, optimizer, shorty)
    elif params.model_fname in ['ECLARE++']:
        return mw.ModelECLAREpp(params, net, criterion, optimizer, shorty)
    return mw.ModelECLARE(params, net, criterion, optimizer, shorty)


def main(params):
    """
        Main function
    """
    torch.manual_seed(params.tree_idx+1)
    torch.cuda.manual_seed_all(params.tree_idx+1)
    np.random.seed(params.tree_idx+1)
    params.label_padding_index = params.num_labels
    net = construct_network(params)
    embeddings = load_embeddings(params)
    net.initialize_embeddings(embeddings)
    print("Model parameters: ", params)
    if params.mode == 'train':
        # NOTE: Use last index as padding label
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optimizer_utils.Optimizer(
            opt_type=params.optim, momentum=params.momentum,
            freeze_embeddings=params.freeze_embeddings,
            weight_decay=params.weight_decay)
        model = construct_model(params, net, criterion, optimizer, None)
        model.transfer_to_devices()
        train(model, params)

    elif params.mode == 'predict':
        model = construct_model(params, net, None, None, None)
        model.transfer_to_devices()
        inference(model, params)

    elif params.mode == 'extract':
        model = construct_model(params, net, None, None, None)
        model.transfer_to_devices()
        extract(model, params)

    else:
        raise NotImplementedError("Unknown mode!")


if __name__ == '__main__':
    args = parameters.Parameters("Parameters")
    args.parse_args()
    main(args.params)
