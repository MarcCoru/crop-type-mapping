import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
import numpy

from torch.autograd import Variable

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

class ConvShapeletModel(nn.Module, BaseEstimator):
    """Convolutional variant of the Learning Time-Series Shapelets model.
    Convolutional variant means that the local distance computations are
    replaced, here, by dot products.


    Learning Time-Series Shapelets was originally presented in [1]_.

    Parameters
    ----------
    n_shapelets_per_size: dict (optional, default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
        None should be used only if `load_from_disk` is set
    num_layers: int (optional, default: 1)
        number of convolutional layers that each convolve the input once.
        num_layers will be ignored if n_shapelets_per_size is specified
    num_hidden: int (optional, default: 50)
        number of hidden dinemnsions (number of convolutional kernels) per layers.
        all layers have the same number of hidden dims.
        If differnet hidden dims per layer are necessary use n_shapelets_per_size
        num_hidden will be ignored if n_shapelets_per_size is specified
    ts_dim: int (optional, default: None)
        Dimensionality (number of modalities) of the time series considered
        None should be used only if `load_from_disk` is set
    n_classes: int (optional, default: None)
        Number of classes in the classification problem
        None should be used only if `load_from_disk` is set
    load_from_disk: str or None (optional, default: None)
        If not None, the model is built from the path given
    use_time_as_feature: bool (optional, default: True)
        insert the time index as additional feature to the input data.

    Note
    ----
        This implementation requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, d=1, n_blobs=2)
    >>> clf = ConvShapeletModel(n_shapelets_per_size={10: 5, 5:3}, ts_dim=1, n_classes=2, epochs=1, verbose=False)
    >>> shapelets = clf.fit(X, y).get_shapelets()
    >>> len(shapelets)
    2
    >>> shapelets[1].shape  # Sorted by increasing shapelet sizes
    (5, 10, 1)
    >>> clf.predict(X).shape
    (40,)
    >>> clf.transform(X).shape
    (40, 8)

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self,
                 num_layers=1,
                 hidden_dims=50,
                 n_shapelets_per_size=None,
                 ts_dim=50,
                 n_classes=None,
                 load_from_disk=None,
                 use_time_as_feature=False,
                 drop_probability=0.5,
                 seqlength=100,
                 scaleshapeletsize=True
                 ):

        super(ConvShapeletModel, self).__init__()
        self.X_fit_ = None
        self.y_fit_ = None
        self.use_time_as_feature = use_time_as_feature

        self.seqlength = seqlength
        self.scaleshapeletsize = scaleshapeletsize

        # batchnormalization after convolution
        self.batchnorm_module = nn.BatchNorm1d(hidden_dims*num_layers)

        # dropout
        self.dropout_module = nn.Dropout(drop_probability)

        if use_time_as_feature:
            ts_dim += 1 # time index as additional input

        if n_shapelets_per_size is None:
            n_shapelets_per_size = build_n_shapelet_dict(num_layers=num_layers, hidden_dims=hidden_dims)

        if load_from_disk is not None:
            self.verbose = True
            self.load(load_from_disk)
        else:
            self.n_shapelets_per_size = n_shapelets_per_size
            self.ts_dim = ts_dim
            self.n_classes = n_classes

            self._set_layers_and_optim()

    def _set_layers_and_optim(self):
        self.shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        if self.scaleshapeletsize:
            [int(size / 100 * self.seqlength) for size in self.shapelet_sizes]

        self.shapelet_blocks = self._get_shapelet_blocks()
        self.logreg_layer = nn.Linear(self.n_shapelets, self.n_classes)
        self.decision_layer = nn.Linear(self.n_shapelets, 1)

    def _get_shapelet_blocks(self):
        return nn.ModuleList([
            nn.Conv1d(in_channels=self.ts_dim,
                      out_channels=self.n_shapelets_per_size[shapelet_size],
                      kernel_size=shapelet_size,
                      padding=shapelet_size//2)
            for shapelet_size in self.shapelet_sizes
        ])

    def _temporal_pooling(self, x):
        pool_size = x.size(-1)
        pooled_x = nn.MaxPool1d(kernel_size=pool_size)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def _features(self, x):
        features_maxpooled = []
        for shp_sz, block in zip(self.shapelet_sizes, self.shapelet_blocks):
            f = block(x)
            f_maxpooled = list()
            for t in range(1, f.shape[2]):
                f_maxpooled.append(self._temporal_pooling(f[:,:,:t]))
            f_maxpooled = torch.stack(f_maxpooled, dim=1)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

    def loss_cross_entropy(self, inputs, targets):
        logits, pts = self._logits(inputs.transpose(1,2))
        logprobabilities = F.log_softmax(logits, dim=-1)

        batchsize, n_times, n_features = logprobabilities.shape

        loss = F.nll_loss(logprobabilities.view(batchsize*n_times,n_features), targets.view(batchsize*n_times))

        stats = dict(
            loss=loss,
        )

        return loss, logprobabilities, pts, stats

    def early_loss_linear(self, inputs, targets, alpha=None, entropy_factor=0):
        return self.loss_cross_entropy(inputs, targets)

    def _init_params(self):
        if self.init_shapelets is not None:
            self.set_shapelets(self.init_shapelets)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if self.init_shapelets is None:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, -1, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, -1, 1)

    @property
    def n_shapelets(self):
        return sum(self.n_shapelets_per_size.values())

    def attentionbudget(self, deltas):

        budget = torch.ones(deltas.shape)
        if torch.cuda.is_available():
            budget = budget.cuda()

        pts = list()
        for t in range(1,deltas.shape[1]):
            pt = deltas[:,t] * budget[:,t-1]
            budget[:,t] = budget[:,t-1] - pt
            pts.append(pt)

        # last time
        pt = budget[:,-1]
        budget[:, -1] = budget[:, -1] - pt
        pts.append(pt)

        return torch.stack(pts,dim=-1), budget

    def _batchnorm(self, x):
        x = x.transpose(2, 1)
        x = self.batchnorm_module(x)
        return x.transpose(2, 1)

    def _logits(self, x):
        if self.use_time_as_feature:
            x = add_time_feature_to_input(x)

        shapelet_features = self._features(x)
        shapelet_features = self._batchnorm(shapelet_features)
        shapelet_features = self.dropout_module(shapelet_features)

        logits = self.logreg_layer(shapelet_features)
        deltas = self.decision_layer(shapelet_features)
        deltas = F.softmax(deltas.squeeze(-1),dim=1)
        pts, budget = self.attentionbudget(deltas)
        return logits, pts

    def forward(self, x, temperature=1):
        logits, deltas = self._logits(x)
        return nn.Softmax(dim=-1)(logits / temperature), deltas

    def predict(self, logprobabilities, Pts):
        return logprobabilities[:,-1,:].argmax(-1)

    def get_shapelets(self):
        shapelets = []
        for block in self.shapelet_blocks:
            weights = block.weight.data.numpy()
            shapelets.append(numpy.transpose(weights, (0, 2, 1)))
        return shapelets

    def set_shapelets(self, l_shapelets):
        """Set shapelet values.

        Parameters
        ----------
        l_shapelets: list of Tensors
            list of Tensors representing the shapelets for each shapelet size,
            sorted by increasing shapelet size

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, d=1, n_blobs=2)
        >>> shp_sz10  = torch.zeros([5, 10], dtype=torch.float32)
        >>> shp_sz5  = torch.zeros([3, 5], dtype=torch.float32)
        >>> clf = ConvShapeletModel(n_shapelets_per_size={10: 5, 5:3}, ts_dim=1, n_classes=2, epochs=1, verbose=False, init_shapelets=[shp_sz5, shp_sz10])
        >>> _ = clf.fit(X, y)
        """
        for shp, block in zip(l_shapelets, self.shapelet_blocks):
            block.weight.data = shp.view(block.weight.shape)

    def save(self, path="model.pth"):
        print("Saving model to " + path)
        params = self.get_params()
        params["model_state"] = self.state_dict()
        params["X_fit_"] = self.X_fit_
        params["y_fit_"] = self.y_fit_
        torch.save(params, path)

    def load(self, path):
        print("Loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.X_fit_ = snapshot.pop('X_fit_', snapshot)
        self.y_fit_ = snapshot.pop('y_fit_', snapshot)
        self.set_params(**snapshot)  # For hyper-parameters

        self._set_layers_and_optim()
        self.load_state_dict(model_state)
        #self.optimizer.load_state_dict(optimizer_state)
        self.eval()  # If at some point we wanted to use batchnorm or dropout

        return snapshot

def add_time_feature_to_input(x):
    """
    adds an additional time feature as additional input dimension.
    the time feature increases with sequence length from zero to one

    :param x: input tensor of dimensions (batchsize, ts_dim, ts_len)
    :return: expanded output tensor of dimensions (batchsize, ts_dim+1, ts_len)
    """

    batchsize, ts_dim, ts_len = x.shape
    # create range
    time_feature = torch.arange(0., float(ts_len)) / ts_len
    # repeat for each batch element
    time_feature = time_feature.repeat(batchsize, 1, 1)
    # move to GPU if available
    if torch.cuda.is_available():
        time_feature = time_feature.cuda()
    # append time_feature to x
    return torch.cat([x, time_feature], dim=1)

def build_n_shapelet_dict(num_layers, hidden_dims):
    """
    Builds a dictionary of format {<kernel_length_in_percentage_of_T>:<num_hidden_dimensions> , ...}
    returns n shapelets per size
    e.g., {10: 100, 20: 100, 30: 100, 40: 100}
    """
    n_shapelets_per_size = dict()
    for layer in range(num_layers):
        shapelet_width = (layer + 1) * 10  # in 10% increments of sequencelength percantage: 10% 20% 30% etc.
        n_shapelets_per_size[shapelet_width] = hidden_dims
    return n_shapelets_per_size