import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
import numpy

from torch.autograd import Variable

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

def tslearn2torch(X, y=None):
    X_ = torch.Tensor(numpy.transpose(X, (0, 2, 1)))
    if y is None:
        return X_

    classes = sorted(list(set(y)))
    y_ = numpy.zeros((X_.shape[0], ))
    for i, v in enumerate(classes):
        y_[y == v] = i
    y_ = torch.Tensor(y_).type(torch.int64)
    z = torch.zeros(len(y_), len(numpy.bincount(y_)))
    y_ = z.scatter_(1, torch.tensor([[i] for i in y_]), 1)
    return X_, y_

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
    ts_dim: int (optional, default: None)
        Dimensionality (number of modalities) of the time series considered
        None should be used only if `load_from_disk` is set
    n_classes: int (optional, default: None)
        Number of classes in the classification problem
        None should be used only if `load_from_disk` is set
    lr: float (optional, default: 0.01)
        Learning rate
    epochs: int (optional, default: 500)
        Number of training epochs
    batch_size: int (optional, default: 64)
        Batch size for training procedure
    lambda_w: float (optional, default: 0.01)
        L2 regularizer weight in the optimized loss
    distillation: boolean (optional, default: False)
        Either a distilled model should be trained.
    temperature: float (optional, default: 1.)
        Temperature parameter of the softmax function for the training phase.
        If distillation==True, the same temperature is used to train both the
        cumbersome and the distilled models.
        Do not touch that parameter unless you are sure of what you do!
    ratio_loss_soft: float (optional, default: .9)
        Only used if distillation==True. Ratio of the soft-target part in the
        loss function where the total loss is:
        ratio_loss_soft * L_soft + (1 - ratio_loss_soft) * L_hard / T^2
        where T is the temperature of the model
    adv_eps: float (optional, default: None)
        Epsilon parameter for the adversarial examples generation.
        If None, no adversarial examples are included.
    init_shapelets: list of numpy arrays (optional, default: None)
        Initial shapelets to be used. If None, shapelets are drawn from a
        standard normal distribution.
        Otherwise, each element in the list should be an array of shape
        (n_shp, shp_sz, ts_dim) where shp_sz is a
        shapelet size for which there should be n_shp in the model.
    verbose: boolean (optional, default: True)
        Should verbose mode be activated
    save_summaries_folder: str or None (optional, default: None)
        If not None, TensorBoard summary is saved in the given folder
    load_from_disk: str or None (optional, default: None)
        If not None, the model is built from the path given
    warm_start: boolean (optional, default: False)
        If False, shapelets are re-initialized at each call to fit.
        If True, shapelets are initialized at the first call to fit and any
        subsequent call to fit will use the already-learned shapelets as
        starting point.

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
                 n_shapelets_per_size=None,  # dict sz_shp -> n_shp
                 ts_dim=50,
                 n_classes=None,
                 load_from_disk=None,
                 ):
        super(ConvShapeletModel, self).__init__()
        self.X_fit_ = None
        self.y_fit_ = None

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
        self.shapelet_blocks = self._get_shapelet_blocks()
        self.logreg_layer = nn.Linear(self.n_shapelets, self.n_classes)

    def _get_shapelet_blocks(self):
        return nn.ModuleList([
            nn.Conv1d(in_channels=self.ts_dim,
                      out_channels=self.n_shapelets_per_size[shapelet_size],
                      kernel_size=shapelet_size)
            for shapelet_size in self.shapelet_sizes
        ])

    def _temporal_pooling(self, x, shapelet_size):
        pool_size = x.size(-1)
        pooled_x = nn.MaxPool1d(kernel_size=pool_size)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def _features(self, x):
        features_maxpooled = []
        for shp_sz, block in zip(self.shapelet_sizes, self.shapelet_blocks):
            f = block(x)
            f_maxpooled = self._temporal_pooling(f, shp_sz)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

    def loss_cross_entropy(self, inputs, targets):
        logits = self._logits(inputs.transpose(1,2))

        logprobabilities = F.log_softmax(logits, dim=1)

        b, t, d = inputs.shape

        Pts = torch.ones([b,t])/t

        loss = F.nll_loss(logprobabilities, targets[:,0])

        stats = dict(
            loss=loss,
        )

        return loss, logprobabilities, Pts, stats

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

    def _logits(self, x):
        shapelet_features = self._features(x)
        logits = self.logreg_layer(shapelet_features)
        return logits

    def forward(self, x, temperature=1):
        logits = self._logits(x)
        return nn.Softmax(dim=-1)(logits / temperature)

    def predict(self, logprobabilities, Pts):
        return logprobabilities.argmax(1)

    def transform(self, X):
        X_ = tslearn2torch(X)
        return self._features(X_).detach().numpy()

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
        self.optimizer.load_state_dict(optimizer_state)
        self.eval()  # If at some point we wanted to use batchnorm or dropout

        return snapshot


