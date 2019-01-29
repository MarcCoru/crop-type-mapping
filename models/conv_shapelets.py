import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
import numpy
import os
from models.attentionbudget import attentionbudget
from models.predict import predict
from models.loss_functions import early_loss_linear, early_loss_cross_entropy, loss_cross_entropy

from torch.autograd import Variable

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

class ConvShapeletModel(nn.Module, BaseEstimator):

    def __init__(self,
                 num_layers=1,
                 hidden_dims=50,
                 n_shapelets_per_size=None,
                 ts_dim=50,
                 n_classes=2,
                 load_from_disk=None,
                 use_time_as_feature=False,
                 drop_probability=0.5,
                 seqlength=100,
                 scaleshapeletsize=True,
                 shapelet_width_increment=10
                 ):

        super(ConvShapeletModel, self).__init__()
        self.X_fit_ = None
        self.y_fit_ = None
        self.use_time_as_feature = use_time_as_feature

        self.seqlength = seqlength
        self.scaleshapeletsize = scaleshapeletsize

        # dropout
        self.dropout_module = nn.Dropout(drop_probability)

        if use_time_as_feature:
            ts_dim += 1 # time index as additional input

        if n_shapelets_per_size is None:
            n_shapelets_per_size = build_n_shapelet_dict(num_layers=num_layers,
                                                         hidden_dims=hidden_dims,
                                                         width_increments=shapelet_width_increment)

        # batchnormalization after convolution
        self.batchnorm_module = nn.BatchNorm1d(sum(n_shapelets_per_size.values()))

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
        torch.nn.init.normal_(self.decision_layer.bias, mean=-1e1, std=1e-1)

    def _get_shapelet_blocks(self):
        return nn.ModuleList([
            ShapeletConvolution(ts_dim=self.ts_dim,
                                shapelet_size=shapelet_size,
                                n_shapelets_per_size=self.n_shapelets_per_size[shapelet_size],
                                )
            #nn.ConstantPad1d((shapelet_size,0),0),
            #nn.Conv1d(in_channels=self.ts_dim,
            #          out_channels=self.n_shapelets_per_size[shapelet_size],
            #          kernel_size=shapelet_size)
                      #padding=shapelet_size) # <- padding of the full shapelet size to make sure that we not use samples from the "future" at pooling time t//2
            for shapelet_size in self.shapelet_sizes
        ])

    def _temporal_pooling(self, x):
        pool_size = x.size(-1)
        pooled_x = nn.MaxPool1d(kernel_size=pool_size)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def _features(self, x):
        sequencelength = x.shape[2]

        features_maxpooled = []
        for shp_sz, block in zip(self.shapelet_sizes, self.shapelet_blocks):
            f = block(x)
            f_maxpooled = list()
            # sequencelength is not equal f.shape[2] -> f is based on padded input
            # -> padding influences length -> we take :sequencelength to avoid using inputs from the future at time t
            for t in range(1, sequencelength+1): # sequencelen
                f_maxpooled.append(self._temporal_pooling(f[:,:,:t]))
            f_maxpooled = torch.stack(f_maxpooled, dim=1)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

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
        deltas = self.decision_layer(torch.sigmoid(shapelet_features))
        deltas = torch.sigmoid(deltas.squeeze(-1))
        pts, budget = attentionbudget(deltas)
        return logits, deltas, pts, budget

    def forward(self, x):
        logits, deltas, pts, budget = self._logits(x)
        logprobabilities = F.log_softmax(logits, dim=2)
        return logprobabilities, deltas, pts, budget

    @torch.no_grad()
    def predict(self, logprobabilities, deltas):
        return predict(logprobabilities, deltas)

    def get_shapelets(self):
        shapelets = []
        for block in self.shapelet_blocks:
            weights = block.weight.data.numpy()
            shapelets.append(numpy.transpose(weights, (0, 2, 1)))
        return shapelets

    def set_shapelets(self, l_shapelets):

        for shp, block in zip(l_shapelets, self.shapelet_blocks):
            block.weight.data = shp.view(block.weight.shape)

    def save(self, path="model.pth",**kwargs):
        print("Saving model to " + path)
        params = self.get_params()
        params["model_state"] = self.state_dict()
        params["X_fit_"] = self.X_fit_
        params["y_fit_"] = self.y_fit_
        # merge kwargs in params
        data = dict(
            params=params,
            config=kwargs
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)

    def load(self, path):
        print("Loading model from " + path)
        data = torch.load(path, map_location="cpu")
        snapshot = data["params"]
        config = data["config"]
        model_state = snapshot.pop('model_state', snapshot)
        self.X_fit_ = snapshot.pop('X_fit_', snapshot)
        self.y_fit_ = snapshot.pop('y_fit_', snapshot)
        self.set_params(**snapshot)  # For hyper-parameters

        self._set_layers_and_optim()
        self.load_state_dict(model_state)
        #self.optimizer.load_state_dict(optimizer_state)
        self.eval()  # If at some point we wanted to use batchnorm or dropout

        for k,v in config.items():
            snapshot[k] = v

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

def build_n_shapelet_dict(num_layers, hidden_dims, width_increments=10):
    """
    Builds a dictionary of format {<kernel_length_in_percentage_of_T>:<num_hidden_dimensions> , ...}
    returns n shapelets per size
    e.g., {10: 100, 20: 100, 30: 100, 40: 100}
    """
    n_shapelets_per_size = dict()
    for layer in range(num_layers):
        shapelet_width = (layer + 1) * width_increments  # in 10 feature increments of sequencelength percantage: 10 20 30 etc.
        n_shapelets_per_size[shapelet_width] = hidden_dims
    return n_shapelets_per_size

class ShapeletConvolution(nn.Module):
    """
    performs left side padding on the input and a convolution
    """
    def __init__(self, shapelet_size, ts_dim, n_shapelets_per_size):
        super(ShapeletConvolution, self).__init__()

        # pure left padding to align classification time t with right edge of convolutional kernel
        self.pad = nn.ConstantPad1d((shapelet_size, 0), 0)
        self.conv = nn.Conv1d(in_channels=ts_dim,
                  out_channels=n_shapelets_per_size,
                  kernel_size=shapelet_size)

    def forward(self, x):
        padded = self.pad(x)
        return self.conv(padded)
