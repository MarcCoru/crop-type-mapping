import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTM
from models import AttentionModule

class AttentionRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dims=3, input_size=(1,1), kernel_size=(1,1), nclasses=5, num_rnn_layers=1, dropout=.8):
        super(AttentionRNN, self).__init__()

        self.nclasses=nclasses

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_rnn_layers, bias=False,
                            batch_first=True, dropout=dropout)

        self.attention = AttentionModule.Attention(hidden_dims)

        self.bn_outputs = nn.BatchNorm1d(hidden_dims)
        self.bn_query = nn.BatchNorm1d(hidden_dims)
        self.bn_class = nn.BatchNorm1d(hidden_dims)

        self.linear_class = nn.Linear(hidden_dims, nclasses, bias=True)

        #self.conv2d_class = nn.Conv2d(in_channels=hidden_dim, out_channels=nclasses, kernel_size=kernel_size,
        #                              padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        #                              bias=True)

    def forward(self,x):

        outputs, last_state_list = self.lstm.forward(x)

        b, t, d = outputs.shape
        o_ = outputs.view(b, -1, d).permute(0, 2, 1)
        outputs = self.bn_outputs(o_).permute(0, 2, 1).view(b, t, d)

        h, c = last_state_list

        query = c[-1]

        query = self.bn_query(query)

        output, weights = self.attention(query.unsqueeze(1),outputs)

        output = self.bn_class(output.squeeze(1))

        logits_class = self.linear_class.forward(output)

        return logits_class, weights.squeeze(1)

    def loss_cross_entropy(self, inputs, targets):
        predicted_logits, weights = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        loss = F.nll_loss(logprobabilities, targets[:,0])

        stats = dict(
            loss=loss,
            )

        return loss, logprobabilities, weights, stats

    def early_loss_linear(self, inputs, targets, alpha=None):
        print("No early_loss_linear implemented in AttentionRNN will use simple cross entropy!")
        return self.loss_cross_entropy(inputs, targets)

    def predict(self, logprobabilities, Pts):
        return logprobabilities.argmax(1)

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
