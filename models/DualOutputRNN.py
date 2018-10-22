import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTMCell, ConvLSTM
import numpy as np
from utils.classmetric import ClassMetric
from utils.logger import Printer
from utils.UCR_Dataset import DatasetWrapper, UCRDataset

# debug
import matplotlib.pyplot as plt

class DualOutputRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3, input_size=(1,1), kernel_size=(1,1,1), nclasses=5, num_rnn_layers=1):
        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses

        self.convlstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size=(kernel_size[1],kernel_size[2]), num_layers=num_rnn_layers,
                 batch_first=True, bias=True, return_all_layers=False)

        self.conv3d_class = nn.Conv3d(in_channels=hidden_dim, out_channels=nclasses, kernel_size=kernel_size,
                                      padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2), bias=True)

        self.conv3d_dec = nn.Conv3d(in_channels=hidden_dim, out_channels=1, kernel_size=kernel_size,
                                      padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                                      bias=True)

    def forward(self,x):

        layer_output_list, last_state_list = self.convlstm.forward(x)
        outputs = layer_output_list[-1]
        #last_hidden, last_state = last_state_list[-1]

        logits_class = self.conv3d_class.forward(outputs.permute(0, 2, 1, 3, 4))
        logits_dec = self.conv3d_dec.forward(outputs.permute(0, 2, 1, 3, 4))
        proba_dec = torch.sigmoid(logits_dec).squeeze(1)

        Pts = list()
        proba_not_decided_yet = list([1.])
        T = proba_dec.shape[1]
        for t in range(T):

            # Probabilities
            if t < T - 1:
                Pt = proba_dec[:,t] * proba_not_decided_yet[-1]
                #proba_not_decided_yet.append(proba_not_decided_yet[-1] * (1.0 - proba_dec))
                proba_not_decided_yet.append(proba_not_decided_yet[-1] - Pt)
            else:
                Pt = proba_not_decided_yet[-1]
                proba_not_decided_yet.append(0.)
            Pts.append(Pt)
        Pts = torch.stack(Pts, dim = 1)

        # stack the lists to new tensor (b,d,t,h,w)
        return logits_class, Pts

    def alphat(self, earliness_factor, out_shape):
        """
        equivalent to _get_loss_earliness

        -> elementwise tensor multiplies the earliness factor with the time (obtained from the expected output shape)
        """

        N, T, H, W = out_shape

        t_vector = torch.arange(T).type(torch.FloatTensor)

        alphat =  ((earliness_factor * torch.ones(*out_shape)).permute(0,3,2,1) * t_vector).permute(0,3,2,1)

        if torch.cuda.is_available():
            return alphat.cuda()
        else:
            return alphat

    def loss(self, inputs, targets, earliness_factor=None):
        predicted_logits, Pts = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        if earliness_factor is not None:
            loss_classification = Pts * F.cross_entropy(predicted_logits, targets.unsqueeze(-1), reduction="none")
            loss_earliness = Pts * self.alphat(earliness_factor, Pts.shape)

            loss = (loss_classification + loss_earliness).sum(dim=1).mean()
        else:
            loss = F.nll_loss(logprobabilities, targets)

        return loss, logprobabilities

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

def plot_Pts(Pts):
    plt.plot(Pts[0, :, 0, 0].detach().numpy())
    plt.show()

def plot_probas(predicted_probas):
    plt.plot(predicted_probas[0, :, :, 0, 0].exp().detach().numpy())
    plt.show()

def print_stats(epoch, stats):
    out_str = "[End of training] Epoch {}: ".format(epoch)
    for k,v in stats.items():
        if len(np.array(v))>0:
            out_str+="{}:{}".format(k,np.array(v).mean())

    print(out_str)

if __name__ == "__main__":
    from tslearn.datasets import CachedDatasets

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    #X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("ElectricDevices")

    nclasses = len(set(y_train))

    model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=64)

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
