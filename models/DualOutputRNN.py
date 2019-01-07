import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from models.loss_functions import early_loss_linear, early_loss_cross_entropy, loss_cross_entropy

def entropy(p):
    return -(p*torch.log(p)).sum(1)

class DualOutputRNN(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=1, dropout=0.2):

        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_rnn_layers, bias=False, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dims)

        self.linear_class = nn.Linear(hidden_dims,nclasses, bias=True)
        self.linear_dec = nn.Linear(hidden_dims, 1, bias=True)

        torch.nn.init.normal_(self.linear_dec.bias, mean=-1e1, std=1e-1)


    def forward(self,x):

        outputs, last_state_list = self.lstm.forward(x)

        b,t,d = outputs.shape
        o_ = outputs.view(b, -1, d).permute(0,2,1)
        outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d)

        logits_class = self.linear_class.forward(outputs)
        logits_dec = self.linear_dec.forward(outputs)

        proba_dec = torch.sigmoid(logits_dec).squeeze(2)

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

    def early_loss_linear(self, inputs, targets, alpha=None, entropy_factor=0):
        """
        Uses linear 1-P(actual class) loss. and the simple time regularization t/T
        L = (1-y\hat{y}) - t/T
        """
        predicted_logits, Pts = self.forward(inputs)
        return early_loss_linear(predicted_logits, Pts, targets, alpha, entropy_factor)

    def early_loss_cross_entropy(self, inputs, targets, alpha=None, entropy_factor=0):
        """
        Uses linear 1-P(actual class) loss. and the simple time regularization t/T
        L = (1-y\hat{y}) - t/T
        """
        predicted_logits, pts = self.forward(inputs)
        return early_loss_cross_entropy(predicted_logits, pts, targets, alpha, entropy_factor)

    def loss_cross_entropy(self, inputs, targets):

        predicted_logits, pts = self.forward(inputs)
        return loss_cross_entropy(predicted_logits, pts, targets)

        logprobabilities = F.log_softmax(predicted_logits, dim=2)

        b,t,c = logprobabilities.shape
        loss = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))

        stats = dict(
            loss=loss,
        )

        return loss, logprobabilities, Pts ,stats


    def predict(self, logprobabilities, Pts):
        """
        Get predicted class labels where Pts is highest
        """
        b, t, c = logprobabilities.shape
        t_class = Pts.argmax(1)  # [c]
        eye = torch.eye(t).type(torch.ByteTensor)

        if torch.cuda.is_available():
            eye = eye.cuda()

        prediction_all_times = logprobabilities.argmax(2)
        prediction_at_t = torch.masked_select(prediction_all_times, eye[t_class])
        return prediction_at_t

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
