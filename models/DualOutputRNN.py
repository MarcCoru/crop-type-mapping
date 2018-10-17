import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTMCell
import numpy as np
from utils.classmetric import ClassMetric
from utils.logger import Printer
from utils.UCR_Dataset import DatasetWrapper, UCRDataset

# debug
import matplotlib.pyplot as plt

class DualOutputRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3, input_size=(1,1), kernel_size=(1,1), nclasses=5):
        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses

        # recurrent cell
        self.rnn = ConvLSTMCell(input_size=input_size,input_dim=input_dim,hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True)

        # Classification layer
        self.conv_class = nn.Conv2d(in_channels=hidden_dim,out_channels=nclasses,kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=True)

        # Decision layer
        self.conv_dec = nn.Conv2d(in_channels=hidden_dim,out_channels=1,kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=True)

        #self.cross_entropy = nn.CrossEntropyLoss()
        # initialize bias with low values to high proba_dec values at the beginning
        #torch.nn.init.normal_(self.conv_dec.bias, mean=-10, std=0.5)
        #pass

    def forward(self,x):

        # initialize hidden state
        hidden, state = self.rnn.init_hidden(batch_size=x.shape[0])

        predicted_decs = list()
        predicted_logits = list()
        proba_not_decided_yet = list([1.])
        Pts = list()

        T = x.shape[1]
        for t in range(T):

            # Model
            hidden, state = self.rnn.forward(x[:, t], (hidden, state))
            proba_dec = torch.sigmoid(self.conv_dec(hidden)).squeeze(1) # (n,h,w) <- squeeze removes the depth dimension
            logits_class = self.conv_class(hidden)
            predicted_decs.append(proba_dec)

            # Probabilities
            if t < T - 1:
                Pt = proba_dec * proba_not_decided_yet[-1]
                proba_not_decided_yet.append(proba_not_decided_yet[-1] * (1.0 - proba_dec))
            else:
                Pt = proba_not_decided_yet[-1]
                proba_not_decided_yet.append(0.)
            Pts.append(Pt)

            predicted_logits.append(logits_class)

        # stack the lists to new tensor (b,d,t,h,w)
        return torch.stack(predicted_logits, dim = 2), torch.stack(Pts, dim = 1)

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

    def fit(self,X,y,
            learning_rate=1e-3,
            earliness_factor=1e-3,
            epochs=3,
            workers=0,
            switch_epoch=2,
            batchsize=3):

        traindataset = DatasetWrapper(X, y)
        traindataset = UCRDataset("trace")

        # handles multithreaded batching and shuffling
        traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                      num_workers=workers)

        printer = Printer()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if torch.cuda.is_available():
            self = self.cuda()

        for epoch in range(epochs):

            # builds a confusion matrix
            metric = ClassMetric(num_classes=self.nclasses)

            logged_loss_early=list()
            logged_loss_class=list()

            for iteration, data in enumerate(traindataloader):
                optimizer.zero_grad()

                inputs, targets = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                predicted_logits, Pts = self.forward(inputs)

                logprobabilities = F.log_softmax(predicted_logits,dim=1)
                maxclass = logprobabilities.argmax(1)
                prediction = maxclass.mode(1)[0]

                stats = metric(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

                if epoch < switch_epoch:
                    loss = F.nll_loss(logprobabilities, targets)
                    #loss = F.cross_entropy(predicted_logits, targets)
                    logged_loss_class.append(loss.detach().cpu().numpy())
                else:
                    loss_classification = Pts * F.cross_entropy(predicted_logits, targets, reduction="none")
                    loss_earliness = Pts * self.alphat(earliness_factor, Pts.shape)

                    loss = (loss_classification + loss_earliness).sum(dim=1).mean()
                    logged_loss_early.append(loss.detach().cpu().numpy())

                stats["loss_early"] = np.array(logged_loss_early).mean()
                stats["loss_class"] = np.array(logged_loss_class).mean()

                printer.print(stats, iteration, epoch)

                loss.backward()
                optimizer.step()

    def save(self, path="model.pth"):
        print("saving model to "+path)
        model_state = self.state_dict()
        torch.save(model_state,path)

    def load(self, path):
        print("loading model from "+path)
        model_state = torch.load(path, map_location="cpu")
        self.load_state_dict(model_state)

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
