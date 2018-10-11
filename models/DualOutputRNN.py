import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTMCell
import numpy as np

class DualOutputRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3, input_size=(1,1), kernel_size=(1,1), nclasses=5):
        super(DualOutputRNN, self).__init__()

        # recurrent cell
        self.rnn = ConvLSTMCell(input_size=input_size,input_dim=input_dim,hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True)

        # Classification layer
        self.conv_class = nn.Conv2d(in_channels=hidden_dim,out_channels=nclasses,kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=True)

        # Decision layer
        self.conv_dec = nn.Conv2d(in_channels=hidden_dim,out_channels=1,kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=True)

    def forward(self,x):

        # initialize hidden state
        hidden, state = self.rnn.init_hidden(batch_size=x.shape[0])

        predicted_decs = list()
        predicted_probas = list()
        proba_not_decided_yet = list([1.])
        Pts = list()

        T = x.shape[1]
        for t in range(T):

            # Model
            hidden, state = self.rnn.forward(x[:, t], (hidden, state))
            proba_dec = torch.sigmoid(self.conv_dec(hidden)).squeeze(1) # (n,h,w) <- squeeze removes the depth dimension
            logits_class = self.conv_class(hidden)
            predicted_decs.append(proba_dec)
            predicted_probas.append(F.log_softmax(logits_class, dim=1))

            # Probabilities
            if t < T - 1:
                Pt = proba_dec * proba_not_decided_yet[-1]
                proba_not_decided_yet.append(proba_not_decided_yet[-1] * (1.0 - proba_dec))
            else:
                Pt = proba_not_decided_yet[-1]
                proba_not_decided_yet.append(0.)
            Pts.append(Pt)

        # stack the lists to new tensor (b,t,d,h,w)
        return torch.stack(predicted_probas, dim = 1), torch.stack(Pts, dim = 1)

    def _build_regularization_earliness(self, earliness_factor, out_shape):
        """
        :param earliness_factor: scaling factor for temporal regularization
        :param out_shape: output shape of format (N, T, H, W)
        :return: tensor of outshape representing earliness*t for each pixel
        """

        N, T, H, W = out_shape

        t_vector = torch.arange(T).type(torch.FloatTensor)
        return ((earliness_factor * torch.ones(*out_shape)).permute(0,3,2,1) * t_vector).permute(0,3,2,1)

    def fit(self,X,y,
            learning_rate=1e-3,
            earliness_factor=1e-3,
            epochs=3,
            workers=0,
            switch_epoch=2,
            batchsize=1):


        traindataset = DatasetWrapper(X, y)

        # handles multithreaded batching and shuffling
        traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                      num_workers=workers)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss = torch.nn.NLLLoss(reduction='none')  # reduction='none'

        if torch.cuda.is_available():
            self = self.cuda()
            loss = loss.cuda()

        stats = dict(
            loss=list(),
            loss_earliness=list(),
            loss_classif=list())

        for epoch in range(epochs):

            for iteration, data in enumerate(traindataloader):
                optimizer.zero_grad()

                inputs, targets = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                predicted_probas, Pts = self.forward(inputs)

                loss_classif = torch.mean((Pts * loss(predicted_probas.permute(0, 2, 1, 3, 4), targets)))

                alphat = model._build_regularization_earliness(earliness_factor=earliness_factor, out_shape=Pts.shape)

                if torch.cuda.is_available():
                    alphat = alphat.cuda()

                loss_earliness = torch.mean((Pts * alphat))

                if epoch < switch_epoch:
                    l = loss_classif
                else:
                    l = loss_classif + loss_earliness

                stats["loss"].append(l.detach().cpu().numpy())
                stats["loss_classif"].append(loss_classif.detach().cpu().numpy())
                stats["loss_earliness"].append(loss_earliness.detach().cpu().numpy())

                l.backward()
                optimizer.step()

            print("[End of training] Epoch:", '%04d' % (epoch + 1),
                  "loss={loss:.9f}, loss_classif={loss_classif:.9f}, loss_earliness={loss_earliness:.9f}".format(
                      loss=np.array(stats["loss"]).mean(),
                      loss_classif=np.array(stats["loss_classif"]).mean(),
                      loss_earliness=np.array(stats["loss_earliness"]).mean()
                  )
                  )

    def save(self, path="model.pth"):
        print("saving model to "+path)
        model_state = self.state_dict()
        torch.save(model_state,path)

    def load(self, path):
        print("loading model from "+path)
        model_state = torch.load(path, map_location="cpu")
        self.load_state_dict(model_state)

class DatasetWrapper(torch.utils.data.Dataset):
    """
    A simple wrapper to insert the dataset in the torch.utils.data.DataLoader module
    that handles multi-threaded loading, sampling, batching and shuffling
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return X_train.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.FloatTensor)
        y = torch.from_numpy(np.array([self.y[idx] - 1])).type(torch.LongTensor)

        # add 1d hight and width dimensions and copy y for each time
        return X.unsqueeze(-1).unsqueeze(-1), y.expand(X.shape[0]).unsqueeze(-1).unsqueeze(-1)



if __name__ == "__main__":
    from tslearn.datasets import CachedDatasets

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    nclasses = len(set(y_train))

    model = DualOutputRNN(input_dim=1, nclasses=nclasses)

    #model.fit(X_train, y_train, epochs=50)
    #model.save("/tmp/model_e50.pth")
    model.load("/tmp/model_e50.pth")

    x = torch.from_numpy(X_test[0]).type(torch.FloatTensor)

    # add batch dimension and hight and width
    x = x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    pred, pt = model.forward(x)

    pass
