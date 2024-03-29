import os
import sys
import numpy as np
import _pickle as pickle
from scipy.sparse import lil_matrix
from .dataset_base import DatasetBase
from sklearn.preprocessing import normalize
from .shortlist_handler import ShortlistHandlerSimple


def construct_dataset(data_dir, fname_features, fname_labels, data=None,
                      model_dir='', mode='train', size_shortlist=-1,
                      normalize_features=True, normalize_labels=True,
                      keep_invalid=False, num_centroids=1,
                      feature_type='sparse', num_clf_partitions=1,
                      feature_indices=None, label_indices=None,
                      shortlist_method='static', shorty=None,
                      classifier_type=None, shorty_path=None):

    return DatasetGraph(
        data_dir, fname_features, fname_labels, data, model_dir, mode,
        feature_indices, label_indices, keep_invalid, normalize_features,
        normalize_labels, num_clf_partitions, size_shortlist,
        num_centroids, feature_type, shorty, "Graph", shortlist_method)

class DatasetSparse(DatasetBase):
    """Dataset to load and use XML-Datasets with shortlist
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    num_centroids: int, optional, default=1
        Multiple representations for labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, size_shortlist=-1, num_centroids=1,
                 feature_type='sparse', shortlist_method='static',
                 shorty=None, label_type='sparse', shortlist_in_memory=True,
                 shorty_path=None):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type=label_type)
        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, \
                    consider using dense dataset.")
        self.feature_type = feature_type
        self.num_centroids = num_centroids
        self.num_clf_partitions = num_clf_partitions
        self.shortlist_in_memory = shortlist_in_memory
        self.size_shortlist = size_shortlist
        self.multiple_cent_mapping = None
        self.shortlist_method = shortlist_method
        self.offset = 0
        indices = None
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            indices = self._remove_samples_wo_features_and_labels()

        if not keep_invalid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)

        if shortlist_method == 'simple':
            self.offset += 1
            self.shortlist = ShortlistHandlerSimple(
                self.num_labels, model_dir, num_clf_partitions,
                mode, size_shortlist, num_centroids,
                shortlist_in_memory, self.multiple_cent_mapping)
        else:
            raise NotImplementedError(
                "Unknown shortlist method: {}!".format(shortlist_method))
        self.label_padding_index = self.num_labels

    def _get_ext_head(self, freq, threshold):
        """Get super-head labels i.e. too many positive points
        """
        return np.where(freq >= threshold)[0]

    def _process_labels_train(self, data_obj, _ext_head_threshold):
        """Process labels for train data
            - Remove labels without any training instance
            - Handle multiple centroids
        """
        super()._process_labels_train(data_obj)
        data_obj['ext_head'] = None
        data_obj['multiple_cent_mapping'] = None
        print("Valid labels after processing: ", self.num_labels)
        if self.num_centroids != 1:
            freq = self.labels.frequency()
            self._ext_head = self._get_ext_head(freq, _ext_head_threshold)
            self.multiple_cent_mapping = np.arange(self.num_labels)
            for idx in self._ext_head:
                self.multiple_cent_mapping = np.append(
                    self.multiple_cent_mapping, [idx]*self.num_centroids)
            data_obj['ext_head'] = self._ext_head
            data_obj['multiple_cent_mapping'] = self.multiple_cent_mapping

    def _process_labels_predict(self, data_obj):
        """Process labels for test data
        """
        super()._process_labels_predict(data_obj)
        try:
            self._ext_head = data_obj['ext_head']
            self.multiple_cent_mapping = data_obj['multiple_cent_mapping']
        except KeyError:
            self._ext_head = None
            self.multiple_cent_mapping = None

    def _process_labels_retrain_w_shortlist(self, data_obj,
                                            _ext_head_threshold):
        """Process labels for retrain with shortlist
        Useful for training labels shortlist after OVA training
        """
        super()._process_labels_predict(data_obj)
        if self.num_centroids != 1:
            print("Creating multiple centroid mappings..")
            freq = self.labels.frequency()
            self._ext_head = self._get_ext_head(freq, _ext_head_threshold)
            self.multiple_cent_mapping = np.arange(self.num_labels)
            for idx in self._ext_head:
                self.multiple_cent_mapping = np.append(
                    self.multiple_cent_mapping, [idx]*self.num_centroids)
            data_obj['ext_head'] = self._ext_head
            data_obj['multiple_cent_mapping'] = self.multiple_cent_mapping

    def _process_labels(self, model_dir, _ext_head_threshold=10000):
        """
            Process labels to handle labels without any training instance;
            Handle multiple centroids if required
        """
        data_obj = {}
        fname = os.path.join(
            model_dir, 'labels_params.pkl' if self._split is None else
            "labels_params_split_{}.pkl".format(self._split))
        if self.mode == 'train':
            self._process_labels_train(data_obj, _ext_head_threshold)
            pickle.dump(data_obj, open(fname, 'wb'))
        elif self.mode == 'retrain_w_shortlist':
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_retrain_w_shortlist(
                data_obj, _ext_head_threshold)
            pickle.dump(data_obj, open(fname, 'wb'))
        else:
            data_obj = pickle.load(open(fname, 'rb'))
            self._process_labels_predict(data_obj)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        if not self.use_shortlist:
            return None
        return self.shortlist[index]

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: np.ndarray or tuple
            for dense: np.ndarray
            for sparse: feature indices and their weights
        labels: tuple
            shortlist: label indices in the shortlist
            labels_mask: 1 for relevant; 0 otherwise
            dist: distance (used during prediction only)
        """
        x = self.features[index]
        y = self.get_shortlist(index)
        return x, y


class DatasetDense(DatasetBase):
    """Dataset to load and use XML-Datasets with full output space only
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    feature_type: str, optional, default='sparse'
        sparse or dense features
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, feature_type='sparse',
                 label_type='dense'):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
            Can Support datasets w/o any label
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         feature_type, label_type)
        if self.mode == 'train':
            # Remove samples w/o any feature or label
            self._remove_samples_wo_features_and_labels()
        if not keep_invalid and self.labels._valid:
            # Remove labels w/o any positive instance
            self._process_labels(model_dir)
        self.feature_type = feature_type
        self.partitioner = None
        self.num_clf_partitions = 1
        if self.labels._valid:  # If no labels are provided
            self.num_clf_partitions = num_clf_partitions
        if self.mode == 'train':
            assert self.labels._valid, "Labels can not be None while training."

        # TODO Take care of this select and padding index
        self.label_padding_index = self.num_labels

    def __getitem__(self, index):
        """
            Get features and labels for index
            Args:
                index: for this sample
            Returns:
                features: : non zero entries
                labels: : numpy array

        """
        x = self.features[index]
        y = self.labels[index]
        return x, y


class DatasetGraph(DatasetSparse):
    """Dataset to load and use XML-Datasets with shortlist
    Parameters
    ---------
    data_dir: str
        data files are stored in this directory
    fname_features: str
        feature file (libsvm or pickle)
    fname_labels: str
        labels file (libsvm or pickle)    
    data: dict, optional, default=None
        Read data directly from this obj rather than files
        Files are ignored if this is not None
        Keys: 'X', 'Y'
    model_dir: str, optional, default=''
        Dump data like valid labels here
    mode: str, optional, default='train'
        Mode of the dataset
    feature_indices: np.ndarray or None, optional, default=None
        Train with selected features only
    label_indices: np.ndarray or None, optional, default=None
        Train for selected labels only
    keep_invalid: bool, optional, default=False
        Don't touch data points or labels
    normalize_features: bool, optional, default=True
        Normalize data points to unit norm
    normalize_lables: bool, optional, default=False
        Normalize labels to convert in probabilities
        Useful in-case on non-binary labels
    num_clf_partitions: int, optional, default=1
        Partition classifier in multiple
        Support for multiple GPUs
    num_centroids: int, optional, default=1
        Multiple representations for labels
    feature_type: str, optional, default='sparse'
        sparse or dense features
    shortlist_type: str, optional, default='static'
        type of shortlist (static or dynamic)
    shorty: obj, optional, default=None
        Useful in-case of dynamic shortlist
    label_type: str, optional, default='dense'
        sparse (i.e. with shortlist) or dense (OVA) labels
    shortlist_in_memory: boolean, optional, default=True
        Keep shortlist in memory if True otherwise keep on disk
    """

    def __init__(self, data_dir, fname_features, fname_labels, data=None,
                 model_dir='', mode='train', feature_indices=None,
                 label_indices=None, keep_invalid=False,
                 normalize_features=True, normalize_labels=False,
                 num_clf_partitions=1, size_shortlist=-1, num_centroids=1,
                 feature_type='sparse', shorty=None, label_type='Parabel',
                 shortlist_in_memory=True):
        """
            Expects 'libsvm' format with header
            Args:
                data_file: str: File name for the set
        """
        super().__init__(data_dir, fname_features, fname_labels, data,
                         model_dir, mode, feature_indices, label_indices,
                         keep_invalid, normalize_features, normalize_labels,
                         num_clf_partitions, size_shortlist, num_centroids,
                         feature_type, "simple", shorty,
                         label_type, shortlist_in_memory)

        if self.labels is None:
            NotImplementedError(
                "No support for shortlist w/o any label, \
                    consider using dense dataset.")
        self.place_holder = np.ones((1, 1), dtype=np.int)
        self.data_have_shortlist = False
        self.use_shortlist = False
        # self.features.add_padding()

    def _prep_for_depth(self, use_shortlist, clusters):
        self.use_shortlist = use_shortlist
        self.labels._prep_for_clusters(use_shortlist, clusters)

    def __getitem__(self, index):
        """Get features and labels for index
        Arguments
        ---------
        index: int
            data for this index
        Returns
        -------
        features: np.ndarray or tuple
            for dense: np.ndarray
            for sparse: feature indices and their weights
        labels: tuple
            shortlist: label indices in the shortlist
            labels_mask: 1 for relevant; 0 otherwise
            dist: distance (used during prediction only)
        """
        x = self.features[index]
        if self.mode == "test":
            return x, self.place_holder, self.get_shortlist(index)
        return x, self.labels[index], self.get_shortlist(index)

    def build_shortlist(self, shorty, net):
        if shorty is not None:
            _, _, n_lbs = self.get_stats()
            _, o_lbs = shorty.shape
            mf = net.C[net.depth]
            print("net depth is", net.depth, net.height, n_lbs)
            self.mf = mf
            Xs = np.arange(mf)
            rows = np.concatenate(list(
                map(lambda x: [x]*mf, np.arange(o_lbs))
            ))
            cols = np.concatenate(list(
                map(lambda x: Xs + x*mf, np.arange(o_lbs))
            ))
            if net.depth == net.height - 2:
                print("In leaf in dalaloader")
                cols = net.leaf_hash[cols]
                rows = rows[cols!=n_lbs]
                cols = cols[cols!=n_lbs]
            to_beam = lil_matrix((o_lbs, n_lbs+1))
            to_beam[rows, cols] = 1
            shorty = shorty.dot(to_beam).tocsr()
            self.shortlist.update_shortlist(shorty)

    def get_shortlist(self, index):
        """
            Get data with shortlist for given data index
        """
        if self.shortlist.shortlist is None:
            return None
        return self.shortlist[index]
