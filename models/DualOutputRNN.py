import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convlstm.convlstm import ConvLSTMCell

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

    def fit(self,X,y, lr=1e-3, earliness_factor=1e-3, epochs=3):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # softmax + Negative Log Likelihood (NLL-Loss) = Crossentropy
        loss = torch.nn.NLLLoss(reduction='none')

        #for epoch in range(epochs):
        #    for batch in range(X.shape[0]):
        #i=0

        predicted_probas, Pts = self.forward(X[i])

        # copy a view of y for each time t and permute dims to (b, t, h, w)
        target = y[0].expand(t, b, h, w).permute(1, 0, 2, 3)
        pred = predicted_probas.permute(0,2,1,3,4) # -> (n, c, t, h, w)

        # sum over time and average over batch
        loss_classif = torch.mean((Pts * loss(pred, target)).sum(1))

        alphat = self._build_regularization_earliness(earliness_factor=earliness_factor, out_shape=Pts.shape)
        # average over time and batch
        loss_earliness = torch.mean(Pts * alphat)

        N, T, _, _, _ = Pts.shape

        loss_earliness = torch.mean(Pts * alphat)


        y[i].expand_as(predicted_probas)

if __name__ == "__main__":
    pass
