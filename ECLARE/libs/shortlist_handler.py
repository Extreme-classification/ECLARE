import numpy as np
import _pickle as pickle
import operator
import os
from scipy.sparse import load_npz, hstack, csr_matrix
from xclib.utils import sparse as sp
import scipy.sparse as sx
import pdb


class ShortlistHandlerBase(object):
    """Base class for ShortlistHandler
    - support for partitioned classifier
    - support for multiple representations for labels

    Parameters
    ----------
    num_labels: int
        number of labels
    shortlist:
        shortlist object
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    num_centroids: int, optional, default=1
        #centroids (useful when using multiple rep)
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_labels, model_dir='',
                 num_clf_partitions=1, mode='train',
                 size_shortlist=-1, num_centroids=1,
                 label_mapping=None):
        self.model_dir = model_dir
        self.num_centroids = num_centroids
        self.num_clf_partitions = num_clf_partitions
        self.size_shortlist = size_shortlist
        self.mode = mode
        self.num_labels = num_labels
        self.label_mapping = label_mapping
        # self._create_shortlist(shortlist)
        self._create_partitioner()
        self.label_padding_index = self.num_labels
        if self.num_clf_partitions > 1:
            self.label_padding_index = self.partitioner.get_padding_indices()

    def _create_shortlist(self, shortlist):
        """
            Create structure to hold shortlist
        """
        self.shortlist = shortlist

    def query(self, *args, **kwargs):
        return self.shortlist(*args, **kwargs)

    def _create_partitioner(self):
        """
            Create partiotionar to for splitted classifier
        """
        self.partitioner = None
        if self.num_clf_partitions > 1:
            if self.mode == 'train':
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.save(os.path.join(
                    self.model_dir, 'partitionar.pkl'))
            else:
                self.partitioner = Partitioner(
                    self.num_labels, self.num_clf_partitions,
                    padding=False, contiguous=True)
                self.partitioner.load(os.path.join(
                    self.model_dir, 'partitionar.pkl'))

    def _pad_seq(self, indices, dist):
        _pad_length = self.size_shortlist - len(indices)
        indices.extend([self.label_padding_index]*_pad_length)
        dist.extend([100]*_pad_length)

    def _remap_multiple_representations(self, indices, vals,
                                        _func=min, _limit=1e5):
        """
            Remap multiple centroids to original labels
        """
        indices = np.asarray(
            list(map(lambda x: self.label_mapping[x], indices)))
        _dict = dict({})
        for id, ind in enumerate(indices):
            _dict[ind] = _func(_dict.get(ind, _limit), vals[id])
        indices, values = zip(*_dict.items())
        indices, values = list(indices), list(values)
        if len(indices) < self.size_shortlist:
            self._pad_seq(indices, values)
        return indices, values

    def _adjust_shortlist(self, pos_labels, shortlist, dist, min_nneg=100):
        """
            Adjust shortlist for a instance
            Training: Add positive labels to the shortlist
            Inference: Return shortlist with label mask
        """
        if self.mode == 'train':
            # TODO: Adjust dist as well
            # If number of positives are more than shortlist_size
            if len(pos_labels) > self.size_shortlist:
                _ind = np.random.choice(
                    len(pos_labels), size=self.size_shortlist-min_nneg, replace=False)
                pos_labels = list(operator.itemgetter(*_ind)(pos_labels))
            neg_labels = list(
                filter(lambda x: x not in set(pos_labels), shortlist))
            diff = self.size_shortlist - len(pos_labels)
            labels_mask = [1]*len(pos_labels)
            dist = [2]*len(pos_labels) + dist[:diff]
            shortlist = pos_labels + neg_labels[:diff]
            labels_mask = labels_mask + [0]*diff
        else:
            labels_mask = [0]*self.size_shortlist
            pos_labels = set(pos_labels)
            for idx, item in enumerate(shortlist):
                if item in pos_labels:
                    labels_mask[idx] = 1
        return np.asarray(shortlist), np.asarray(labels_mask), np.asarray(dist)

    def _get_sl_one(self, index, pos_labels):
        if self.shortlist.data_init:
            shortlist, dist = self.query(index)
            # Remap to original labels if multiple centroids are used
            shortlist, labels_mask, dist = self._adjust_shortlist(
                pos_labels, shortlist, dist)
        else:
            shortlist = [0]*self.size_shortlist
            labels_mask = [0]*self.size_shortlist
            dist = [0]*self.size_shortlist
        return shortlist, labels_mask, dist

    def _get_sl_partitioned(self, index, pos_labels):
        # Partition labels
        pos_labels = self.partitioner.split_indices(pos_labels)
        if self.shortlist.data_init:  # Shortlist is initialized
            _shortlist, _dist = self.query(index)
            shortlist, labels_mask, dist, rev_map = [], [], [], []
            # Get shortlist for each classifier
            for idx in range(self.num_clf_partitions):
                __shortlist, __labels_mask, __dist = self._adjust_shortlist(
                    pos_labels[idx],
                    _shortlist[idx].tolist(),
                    _dist[idx].tolist())
                shortlist.append(__shortlist)
                labels_mask.append(__labels_mask)
                dist.append(__dist)
                rev_map += self.partitioner.map_to_original(__shortlist, idx)
        else:  # Shortlist is un-initialized
            shortlist, labels_mask, dist = [], [], []
            for idx in range(self.num_clf_partitions):
                shortlist.append([0]*self.size_shortlist)
                labels_mask.append([0]*self.size_shortlist)
                dist.append([0]*self.size_shortlist)
            rev_map = [0]*self.size_shortlist*self.num_clf_partitions  # Dummy
        return shortlist, labels_mask, dist, rev_map

    def get_shortlist(self, index, pos_labels=None):
        """
            Get data with shortlist for given data index
        """
        if self.num_clf_partitions > 1:
            return self._get_sl_partitioned(index, pos_labels)
        else:
            return self._get_sl_one(index, pos_labels)

    def get_partition_indices(self, index):
        return self.partitioner.get_indices(index)


class ShortlistHandlerSimple(ShortlistHandlerBase):
    """ShortlistHandler with static shortlist
    - save/load/update/process shortlist
    - support for partitioned classifier
    - support for multiple representations for labels
    Parameters
    ----------
    num_labels: int
        number of labels
    model_dir: str, optional, default=''
        save the data in model_dir
    num_clf_partitions: int, optional, default=''
        #classifier splits
    mode: str: optional, default=''
        mode i.e. train or test or val
    size_shortlist:int, optional, default=-1
        get shortlist of this size
    num_centroids: int, optional, default=1
        #centroids (useful when using multiple rep)
    in_memory: bool: optional, default=True
        Keep the shortlist in memory or on-disk
    label_mapping: None or dict: optional, default=None
        map labels as per this mapping
    """

    def __init__(self, num_labels, model_dir='', num_clf_partitions=1,
                 mode='train', size_shortlist=-1, num_centroids=1,
                 in_memory=True, label_mapping=None):
        super().__init__(num_labels, model_dir, num_clf_partitions,
                         mode, size_shortlist, num_centroids, label_mapping)
        self.in_memory = in_memory
        self._create_shortlist()
        print("Simple: This is a simple shortlist handler")

    def query(self, index):
        shorty = self.shortlist[index]
        return shorty

    def _get_sl_one(self, index, pos_labels):
        return self.shortlist[index]

    def _create_shortlist(self):
        """
            Create structure to hold shortlist
        """
        self.shortlist = None

    def update_shortlist(self, shortlist, random_neg=-1):
        """
            Update label shortlist for each instance
        """
        self.shortlist = shortlist
        n_inst, _ = shortlist.shape
        print("Average shortlist size", shortlist.nnz/n_inst)

    def __len__(self):
        return len(self.shortlist)
    
    def __getitem__(self, index):
        return self.shortlist[index]
