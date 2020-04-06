import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os


""" 
DuPLO: A DUal view Point deep Learning architecture
for time series classificatiOn

by Interdonato et al. (2018)

Notes from the paper
https://www.sciencedirect.com/science/article/abs/pii/S0924271619300115

# CNN_branch
Conv 1 3x3 256
Conv 2 3x3 512
Conv 3 1x1 1024

Conv = Conv Relu BatchNorm

# RNN branch

Shallow CNN (SCNN)
    * CNN1 3x3 (?) 32
    * CNN2 3x3 (?) 64
    CNN = Conv + Linear Filter (?) + Relu + BN

GRU (single layer 64 -> 64)

Attention

v = tanh(hW + b)
w = softmax(v*u)
out = sum wh
"""

class DuPLO(torch.nn.Module):
    def __init__(self, input_dim=1, nclasses=5, sequencelength=70, dropout=0.4):
        super(DuPLO, self).__init__()

        self.cnn = CNN(input_dim=input_dim*sequencelength, kernel_size=1, dropout=dropout)

        self.scnn = SCNN(input_dim=input_dim, kernel_size=1, dropout=dropout)
        self.rnn = nn.GRU(input_size=64, hidden_size=1024, num_layers=1,
                            bias=True, batch_first=True, bidirectional=False, dropout=dropout)

        self.attention = SoftAttention(hidden_dim=1024)
        self.outlinear = nn.Linear(in_features=2048,out_features=nclasses)
        self.outlinear_cnn = nn.Linear(in_features=1024, out_features=nclasses)
        self.outlinear_rnn = nn.Linear(in_features=1024, out_features=nclasses)

    def forward(self,x):
        """
        x: time series N x D x T
        """
        N,D,T = x.shape

        # CNN branch_ N x D*T x H=1 x W=1
        x_stacked_image = x.reshape(N, D * T , 1, 1)
        cnn_features = self.cnn(x_stacked_image).squeeze(-1).squeeze(-1)

        # reshape x to process each image separately (treat each time as sample in batch <- N*T)
        x_scnn_stacked = x.transpose(1, 2).reshape(N*T, D, 1, 1)

        # reshape x back to N x T x H
        x_scnn = self.scnn(x_scnn_stacked).view(N, T, 64)
        rnn_output, last_state = self.rnn(x_scnn)

        rnn_features = self.attention(rnn_output)

        features = torch.cat([cnn_features, rnn_features], dim=1)

        logits = self.outlinear(features)
        logits_cnn = self.outlinear_cnn(cnn_features)
        logits_rnn = self.outlinear_rnn(rnn_features)

        return F.log_softmax(logits, dim=-1), F.log_softmax(logits_cnn, dim=-1), F.log_softmax(logits_rnn, dim=-1)

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot


class CNN(torch.nn.Module):
    """
    Conv 1 3x3 256
    Conv 2 3x3 512
    Conv 3 1x1 1024
    """

    def __init__(self, input_dim, kernel_size, dropout):
        super(CNN, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=kernel_size, padding=(kernel_size//2)),
            Conv_Relu_BatchNorm_Dropout(input_dim=128, hidden_dims=256, kernel_size=kernel_size, dropout=dropout),
            Conv_Relu_BatchNorm_Dropout(input_dim=256, hidden_dims=512, kernel_size=kernel_size, dropout=dropout),
            Conv_Relu_BatchNorm_Dropout(input_dim=512, hidden_dims=1024, kernel_size=kernel_size, dropout=dropout)
        )

    def forward(self, X):
        return self.block(X)

class SoftAttention(torch.nn.Module):
    """
    v = tanh(hW + b)
    w = softmax(v*u)
    out = sum wh

    see eqs 5-7 in https://www.sciencedirect.com/science/article/abs/pii/S0924271619300115
    """
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()

        # the linear layer takes care of Wa and ba
        self.linear = nn.Linear(in_features=hidden_dim,out_features=hidden_dim, bias=True)
        self.tanh = nn.Tanh()

        self.ua = nn.Parameter(torch.Tensor(hidden_dim))
        torch.nn.init.normal_(self.ua, mean=0.0, std=0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, *_ = x.shape

        # eq 5
        va = self.tanh(self.linear(x))

        # eq 6
        batchwise_ua = self.ua.repeat(N, 1)
        omega = self.softmax(torch.bmm(va, batchwise_ua.unsqueeze(-1)))

        # eq 7 rnn_feat: N x 64
        rnn_feat = torch.bmm(x.transpose(1, 2), omega).squeeze(-1)

        return rnn_feat

class SCNN(torch.nn.Module):
    def __init__(self, input_dim, kernel_size, dropout=0.4):
        super(SCNN, self).__init__()

        self.block = nn.Sequential(
            Conv_Relu_BatchNorm_Dropout(input_dim=input_dim, hidden_dims=32, kernel_size=kernel_size, dropout=dropout),
            Conv_Relu_BatchNorm_Dropout(input_dim=32, hidden_dims=64, kernel_size=kernel_size, dropout=dropout)
        )

    def forward(self, X):
        return self.block(X)

class Conv_Relu_BatchNorm_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=5, dropout=0.4):
        super(Conv_Relu_BatchNorm_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dims),
            nn.Dropout(dropout)
        )

    def forward(self, X):
        return self.block(X)

if __name__ == '__main__':
    N = 256 # batchsize
    T = 70 # sequencelength
    D = 13 # features
    nclasses = 10
    dropout=0.4


    x = torch.ones(N,D,T).float()
    model = DuPLO(input_dim=D, nclasses=nclasses, sequencelength=T, dropout=dropout)
    logprobabilities = model.forward(x)

