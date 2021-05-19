import numpy as np
import torch
import copy
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize as scale
from functools import partial
import operator
import functools
import _pickle as pik
import tqdm
from multiprocessing import Pool, cpu_count


def _normalize(X, norm='l2'):
    X = scale(X, norm='l2')
    return X


def b_kmeans_dense_multi(fts_lbl, index, metric='cosine', tol=1e-4, leakage=None):
    lbl_cent = _normalize(np.squeeze(fts_lbl[:, 0, :]))
    lbl_fts = _normalize(np.squeeze(fts_lbl[:, 1, :]))
    if lbl_cent.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    _centeroids = lbl_cent[cluster]
    _sim = np.dot(lbl_cent, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            np.mean(lbl_cent[x, :], axis=0) for x in c_lbs
        ]))
        _sim_1 = np.dot(lbl_cent, _centeroids.T)
        _centeroids = _normalize(np.vstack([
            np.mean(lbl_fts[x, :], axis=0) for x in c_lbs
        ]))
        _sim_2 = np.dot(lbl_fts, _centeroids.T)
        _sim = _sim_1 + _sim_2
        old_sim, new_sim = new_sim, np.sum([np.sum(_sim[c_lbs[0], 0]),
                                            np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_dense(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    labels_features = _normalize(labels_features)
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = labels_features[cluster]
    _similarity = np.dot(labels_features, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            np.mean(labels_features[x, :], axis=0) for x in clustered_lbs
        ]))
        _similarity = np.dot(labels_features, _centeroids.T)
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])

    return list(map(lambda x: index[x], clustered_lbs))


def b_kmeans_sparse(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    labels_features = _normalize(labels_features)
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = _normalize(labels_features[cluster].todense())
    _sim = _sdist(labels_features, _centeroids)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            labels_features[x, :].mean(axis=0) for x in c_lbs]))
        _sim = _sdist(labels_features, _centeroids)
        old_sim, new_sim = new_sim, np.sum([
            np.sum(_sim[c_lbs[0], 0]), np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def _sdist(XA, XB, norm=None):
    return XA.dot(XB.transpose())


def _merge_tree(cluster, verbose_label_index, avg_size=0, force=False):
    if cluster[0].size < verbose_label_index[0].size:
        print("Merging trees", np.log2(len(cluster)))
        return cluster + verbose_label_index, [np.asarray([])]
    elif verbose_label_index[0].size > 0 and force:
        if verbose_label_index.shape[0] > 0:
            print("Force Merging trees")
            return cluster + verbose_label_index, [np.asarray([])]
        else:
            print("Nothing else to do")
            return cluster, [np.asarray([])]
    else:
        return cluster, verbose_label_index


def cluster_labels(labels, clusters, verbose_label_index, num_nodes, splitter):
    start = time.time()
    least = min(16, num_nodes)
    clusters, verbose_label_index = _merge_tree(clusters, verbose_label_index)
    while len(clusters) < num_nodes:
        temp_cluster_list = functools.reduce(
            operator.iconcat,
            map(lambda x: splitter(labels[x], x),
                clusters), [])
        end = time.time()
        print("Total clusters {}".format(len(temp_cluster_list)),
              "Avg. Cluster size {}".format(
            np.mean(list(map(len, temp_cluster_list+verbose_label_index)))),
            "Total time {} sec".format(end-start))
        clusters = temp_cluster_list
        clusters, verbose_label_index = _merge_tree(
            clusters, verbose_label_index)
        del temp_cluster_list
    # print(cpu_count()-1)
    # with Pool(6) as p:
    #     while len(clusters) < num_nodes:
    #         temp_cluster_list = functools.reduce(
    #             operator.iconcat,
    #             p.starmap(
    #                 splitter,
    #                 map(lambda cluster: (labels[cluster], cluster),
    #                     clusters)
    #             ), [])
    #         end = time.time()
    #         print("Total clusters {}".format(len(temp_cluster_list)),
    #               "Avg. Cluster size {}".format(
    #                   np.mean(list(map(len, temp_cluster_list+verbose_label_index)))),
    #               "Total time {} sec".format(end-start))
    #         clusters = temp_cluster_list
    #         clusters, verbose_label_index = _merge_tree(
    #             clusters, verbose_label_index)
    #         del temp_cluster_list
    return clusters, verbose_label_index


def representative(lbl_fts):
    scores = np.ravel(np.sum(np.dot(lbl_fts, lbl_fts.T), axis=1))
    return lbl_fts[np.argmax(scores)]


class hash_map_index:
    def __init__(self, clusters, label_to_idx, total_elements, total_valid_nodes, padding_idx=None):
        self.clusters = clusters
        self.padding_idx = padding_idx
        self.total_elements = total_elements
        self.size = total_valid_nodes
        self.weights = None
        if padding_idx is not None:
            self.weights = np.zeros((self.total_elements), np.float)
            self.weights[label_to_idx == padding_idx] = -np.inf

        self.hash_map = label_to_idx

    def _get_hash(self):
        return self.hash_map

    def _get_weights(self):
        return self.weights


class build_tree:
    def __init__(self, b_factors=[2], M=1, method='random',
                 leaf_size=0, force_shallow=True):
        self.b_factors = b_factors
        self.C = []
        self.method = method
        self.leaf_size = leaf_size
        self.force_shallow = force_shallow
        self.height = 2

    def fit(self, label_index=[], verbose_label_index=[], lbl_repr=None):
        self.num_labels = lbl_repr.shape[0]
        clusters = [label_index]
        self.hash_map_array = []
        print("Total verbose labels", verbose_label_index.size)

        if len(lbl_repr.shape) > 2:
            print("Using multi objective kmeans++")
            b_kmeans = b_kmeans_dense_multi

        elif isinstance(lbl_repr, np.ndarray):
            print("Using dense kmeans++")
            b_kmeans = b_kmeans_dense

        else:
            lbl_repr = lbl_repr.tocsr()
            b_kmeans = b_kmeans_sparse

        if self.method == "NoCluster":
            self.height = 1
            print("No need to create splits")
            n_lb = self.num_labels
            self.hash_map_array.append(hash_map_index(
                None, np.concatenate(clusters), n_lb, n_lb, n_lb))
            return

        self._parabel(lbl_repr, clusters, [verbose_label_index],
                      b_kmeans, self.force_shallow)

    def _parabel(self, labels, clusters, verbose_label_index,
                 splitter=None, force_shallow=True):
        depth = 0
        T_verb_lbl = verbose_label_index[0].size
        while True:
            orignal_num_nodes = 2**self.b_factors[depth]
            n_child_nodes = orignal_num_nodes
            if self.num_labels/n_child_nodes < T_verb_lbl or \
                    len(self.b_factors) == 1:
                if T_verb_lbl > 0:
                    add_at = np.floor(np.log2(self.num_labels/T_verb_lbl))+1
                    addition = 2**(self.b_factors[depth]-add_at)
                    n_child_nodes += addition
            depth += 1
            print("Building tree at height %d with nodes: %d" %
                  (depth, n_child_nodes))
            if n_child_nodes >= self.num_labels:
                print("No need to do clustering")
                clusters = list(np.arange(self.num_labels).reshape(-1, 1))
            else:
                clusters, verbose_label_index = cluster_labels(
                    labels, clusters, verbose_label_index,
                    orignal_num_nodes, splitter)
                if depth == len(self.b_factors):
                    clusters, verbose_label_index = _merge_tree(
                        clusters, verbose_label_index, True)
            self.hash_map_array.append(
                hash_map_index(
                    clusters,
                    np.arange(n_child_nodes),
                    n_child_nodes,
                    n_child_nodes
                )
            )
            self.C.append(max(list(map(lambda x: x.size, clusters))))
            if depth == len(self.b_factors):
                print("Preparing Leaf")
                break

        self.height = depth+1
        self.max_node_idx = np.int(n_child_nodes*self.C[-1])
        print("Building tree at height %d with max leafs: %d" %
              (self.height, self.max_node_idx))
        _labels_path_array = np.full(
            (self.max_node_idx), self.num_labels,
            dtype=np.int)
        for idx, c in enumerate(clusters):
            index = np.arange(c.size) + idx*self.C[-1]
            _labels_path_array[index] = clusters[idx]
        self.hash_map_array.append(
            hash_map_index(None,
                           _labels_path_array,
                           self.max_node_idx,
                           self.num_labels,
                           self.num_labels))
        print("Sparsity of leaf is %0.2f" %
              ((1-(self.num_labels/self.max_node_idx))*100))

    def _get_cluster_depth(self, depth):
        return self.hash_map_array[depth].clusters

    def load(self, fname):
        self.__dict__ = pik.load(open(fname, 'rb'))

    def save(self, fname):
        pik.dump(self.__dict__, open(fname, 'wb'))
