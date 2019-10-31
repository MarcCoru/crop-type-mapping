import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from models.ClassificationModel import ClassificationModel
from models.AttentionModule import Attention
from torch.nn.modules.normalization import LayerNorm

SEQUENCE_PADDINGS_VALUE=-1

def entropy(p):
    return -(p*torch.log(p)).sum(1)

class RNN(ClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=1, dropout=0.2, bidirectional=False,
                 use_batchnorm=False, use_attention=False, use_layernorm=True):

        super(RNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.use_layernorm = use_layernorm
        self.bidirectional = bidirectional

        self.d_model = num_rnn_layers*hidden_dims

        if use_layernorm:
            # perform
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims*bidirectional)*num_rnn_layers)
            #self.lstmlayernorm = nn.LayerNorm(hidden_dims)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_rnn_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if bidirectional: # if bidirectional we have twice as many hidden dims after lstm encoding...
            hidden_dims = hidden_dims * 2

        outlineardims = hidden_dims if use_attention else hidden_dims*num_rnn_layers
        self.linear_class = nn.Linear(outlineardims, nclasses, bias=True)

        if use_attention:
            #attention_dims = hidden_dims // 2 if bidirectional else hidden_dims

            self.attention = Attention(hidden_dims, attention_type="dot")

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dims)


    def _logits(self, x):

        # b,d,t -> b,t,d
        x = x.transpose(1,2)

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, last_state_list = self.lstm.forward(x)

        h, c = last_state_list
        if self.use_attention:
            if self.bidirectional:
                query_forward = c[-1]
                query_backward = c[-2]
                query = torch.cat([query_forward, query_backward],1)
            else:
                query = c[-1]

            #query = self.bn_query(query)

            h, weights = self.attention(query.unsqueeze(1), outputs)
            h = h.squeeze(1)
            #outputs, weights = self.attention(outputs, outputs)
        else:
            nlayers, batchsize, n_hidden = c.shape
            # use last cell state as classificaiton features
            h = self.clayernorm(c.transpose(0,1).contiguous().view(batchsize,nlayers*n_hidden))

        logits = self.linear_class.forward(h)

        if self.use_attention:
            pts = weights
        else:
            pts = None

        return logits, None, pts, None

    def forward(self,x):
        logits, deltas, pts, budget = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        # stack the lists to new tensor (b,d,t,h,w)
        return logprobabilities, deltas, pts, budget

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

if __name__ == "__main__":
    from tslearn.datasets import CachedDatasets

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    #X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("ElectricDevices")

    nclasses = len(set(y_train))

    model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dims=64)

    model.fit(X_train, y_train, epochs=100, switch_epoch=50 ,earliness_factor=1e-3, batchsize=75, learning_rate=.01)
    model.save("/tmp/model_200_e0.001.pth")
    model.load("/tmp/model_200_e0.001.pth")

    # add batch dimension and hight and width

    pts = list()

    # predict a few samples
    with torch.no_grad():
        for i in range(100):
            x = torch.from_numpy(X_test[i]).type(torch.FloatTensor)

            x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            pred, pt = model.forward(x)
            pts.append(pt[0,:,0,0].detach().numpy())

    pass
