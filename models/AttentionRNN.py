import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTM
from models import AttentionModule

class AttentionRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3, input_size=(1,1), kernel_size=(1,1), nclasses=5, num_rnn_layers=1, use_batchnorm=True):
        super(AttentionRNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm

        self.convlstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size=(kernel_size[0],kernel_size[1]), num_layers=num_rnn_layers,
                 batch_first=True, bias=not use_batchnorm, return_all_layers=False)

        self.attention = AttentionModule.Attention(hidden_dim)

        if use_batchnorm:
            self.bn_query = nn.BatchNorm1d(hidden_dim)
            self.bn_context = nn.BatchNorm1d(hidden_dim)

        self.conv2d_class = nn.Conv2d(in_channels=hidden_dim, out_channels=nclasses, kernel_size=kernel_size,
                                      padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                                      bias=not use_batchnorm)

    def forward(self,x):

        layer_output_list, last_state_list = self.convlstm.forward(x)
        outputs = layer_output_list[-1]

        h, c = last_state_list[-1]
        # reshape c from [b,d,h,w] to [b,t=1,d]
        query = c.squeeze().unsqueeze(1)


        # bring outputs form [b,t,d,h,w] to [b,t,d] (assuming h,w=1)
        context = outputs.squeeze()

        # bn
        if self.use_batchnorm:
            query = self.bn_query(query.permute(0, 2, 1)).permute(0, 2, 1)
            context = self.bn_context(context.permute(0, 2, 1)).permute(0, 2, 1)

        output, weights = self.attention(query, context)

        # bring output from [b,t=1,d] to [b,d,h=1,w=1]
        output = output.squeeze(1).unsqueeze(-1).unsqueeze(-1)

        logits_class = self.conv2d_class.forward(output)

        # stack the lists to new tensor (b,d,t,h,w)
        return logits_class, weights.permute(0,2,1).unsqueeze(-1)

    def loss(self, inputs, targets, **kwargs):
        predicted_logits, weights = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        loss = F.nll_loss(logprobabilities, targets[:,0,:,:])

        stats = dict(
            loss=loss,
            )

        return loss, logprobabilities.argmax(1), weights, stats

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
