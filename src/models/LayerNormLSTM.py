#!/usr/bin/env python

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

#from .normalize import LayerNorm


# from https://github.com/seba-1511/lstms.pth/blob/master/lstms/lstm.py

class LSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:

    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_method='pytorch'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert (dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)

class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size, learnable=True, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = T(1, input_size).fill_(0)
        self.beta = T(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            W = P
        else:
            W = V
        self.alpha = W(self.alpha)
        self.beta = W(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - th.mean(x, 1).expand_as(x)) / th.sqrt(th.var(x, 1).expand_as(x) + self.epsilon)
        if self.learnable:
            x =  self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)


class LayerNormLSTM(LSTM):

    """
    Layer Normalization LSTM, based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    Special args:
        ln_preact: whether to Layer Normalize the pre-activations.
        learnable: whether the LN alpha & gamma should be used.
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', ln_preact=True, learnable=True):
        super(LayerNormLSTM, self).__init__(input_size=input_size,
                                            hidden_size=hidden_size,
                                            bias=bias,
                                            dropout=dropout,
                                            dropout_method=dropout_method)
        if ln_preact:
            self.ln_i2h = LayerNorm(4*hidden_size, learnable=learnable)
            self.ln_h2h = LayerNorm(4*hidden_size, learnable=learnable)
        self.ln_preact = ln_preact
        self.ln_cell = LayerNorm(hidden_size, learnable=learnable)

    def forward(self, x, hidden):
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
                c_t.data.set_(th.mul(c_t, self.mask).data)
                c_t.data *= 1.0/(1.0 - self.dropout)

        c_t = self.ln_cell(c_t)
        h_t = th.mul(o_t, c_t.tanh())

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                    h_t.data.set_(th.mul(h_t, self.mask).data)
                    h_t.data *= 1.0/(1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        return h_t, (h_t, c_t)
