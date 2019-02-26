import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from models.EarlyClassificationModel import EarlyClassificationModel

SEQUENCE_PADDINGS_VALUE=-1

def entropy(p):
    return -(p*torch.log(p)).sum(1)

class DualOutputRNN(EarlyClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=1, dropout=0.2):

        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_rnn_layers, bias=False, batch_first=True, dropout=dropout, bidirectional=False)
        self.bn = nn.BatchNorm1d(hidden_dims)

        self.linear_class = nn.Linear(hidden_dims,nclasses, bias=True)
        self.linear_dec = nn.Linear(hidden_dims, 1, bias=True)

        torch.nn.init.normal_(self.linear_dec.bias, mean=-1e1, std=1e-1)

    def _logits(self, x):
        batchsize, indim, maxsequencelength = x.shape

        # get sequence lengths from the index of the first padded value
        #lengths = torch.argmax((x[:, 0, :] == SEQUENCE_PADDINGS_VALUE), dim=1)

        # if no padded values insert sequencelength as sequencelength
        #lengths[lengths == 0] = maxsequencelength

        # sort sequences descending to prepare for packing
        #lengths, idxs = lengths.sort(0, descending=True)

        # order x in decreasing seequence lengths
        #x = x[idxs]

        #packed = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(1,2), lengths, batch_first=True)
        outputs, last_state_list = self.lstm.forward(x.transpose(1,2))
        #outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        b,t,d = outputs.shape
        o_ = outputs.view(b, -1, d).permute(0,2,1)
        outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d)

        logits = self.linear_class.forward(outputs)
        deltas = self.linear_dec.forward(outputs)

        deltas = torch.sigmoid(deltas).squeeze(2)

        pts, budget = self.attentionbudget(deltas)

        return logits, deltas, pts, budget

    def forward(self,x):
        logits, deltas, pts, budget = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=2)
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
