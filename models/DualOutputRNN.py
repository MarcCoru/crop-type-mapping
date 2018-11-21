import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.convlstm.convlstm import ConvLSTM

class DualOutputRNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3, input_size=(1,1), kernel_size=(1,1,1), nclasses=5, num_rnn_layers=1, use_batchnorm=True):
        super(DualOutputRNN, self).__init__()

        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm

        self.convlstm = ConvLSTM(input_size, input_dim, hidden_dim, kernel_size=(kernel_size[1],kernel_size[2]), num_layers=num_rnn_layers,
                 batch_first=True, bias=not use_batchnorm, return_all_layers=False)

        self.conv3d_class = nn.Conv3d(in_channels=hidden_dim, out_channels=nclasses, kernel_size=kernel_size,
                                      padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2), bias=True)

        self.conv3d_dec = nn.Conv3d(in_channels=hidden_dim, out_channels=1, kernel_size=kernel_size,
                                      padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                                      bias=True)

        torch.nn.init.normal_(self.conv3d_dec.bias, mean=-1e1, std=1e-1)

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self,x):

        layer_output_list, last_state_list = self.convlstm.forward(x)
        outputs = layer_output_list[-1]
        #last_hidden, last_state = last_state_list[-1]

        if self.use_batchnorm:
            # flatted to [b,d,t], perform bn and reshape to old form
            b,t,d,h,w = outputs.shape
            o_ = outputs.view(b, -1, d).permute(0,2,1)
            outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d,h,w)

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

    def early_loss(self, inputs, targets, alpha=None):
        predicted_logits, Pts = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        b, c, t, h, w = logprobabilities.shape

        #logprobabilities_flat = logprobabilities.view(b * t * h * w, c)
        #targets_flat = targets.view(b * t * h * w)

        # an index at which time the classification took place
        t_index = (torch.ones(b,h,w,t) * torch.arange(t).type(torch.FloatTensor)).transpose(1,3)

        # the reward for early classification 1-(T/t)
        #t_reward = 1 - t_index.repeat(b * h * w) / t
        t_reward = 1 - t_index / t

        eye = torch.eye(c).type(torch.ByteTensor)
        if torch.cuda.is_available():
            eye = eye.cuda()
            t_reward = t_reward.cuda()

        # [b, t, h, w, c]
        targets_one_hot = eye[targets]

        # implement the y*\hat{y} part of the loss function
        # (permute moves classes to last dimenions -> same as targets_one_hot)
        y_haty = torch.masked_select(logprobabilities.permute(0,2,3,4,1), targets_one_hot)
        y_haty = y_haty.view(b, t, h, w).exp()

        #reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
        reward_earliness = (Pts * y_haty * t_reward).sum(1).mean()
        loss_classification = (Pts * F.nll_loss(logprobabilities, targets,reduction='none')).sum(1).mean()

        loss =  alpha * loss_classification - (1 - alpha) * reward_earliness

        stats = dict(
            loss=loss,
            loss_classification=loss_classification,
            reward_earliness=reward_earliness,
        )

        return loss, logprobabilities, Pts ,stats

    def early_loss_simple(self, inputs, targets, alpha=None):
        predicted_logits, Pts = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        b, c, t, h, w = logprobabilities.shape

        # an index at which time the classification took place
        t_index = (torch.ones(b,h,w,t) * torch.arange(t).type(torch.FloatTensor)).transpose(1,3).cuda()

        #reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
        loss_earliness = (Pts*(t_index / t)).sum(1).mean()
        loss_classification = (Pts * F.nll_loss(logprobabilities, targets,reduction='none')).sum(1).mean()

        loss =  alpha * loss_classification + (1 - alpha) * loss_earliness

        stats = dict(
            loss=loss,
            loss_classification=loss_classification,
            loss_earliness=loss_earliness,
        )

        return loss, logprobabilities, Pts ,stats

    def early_loss_linear(self, inputs, targets, alpha=None):
        """
        Uses linear 1-P(actual class) loss. and the simple time regularization t/T
        L = (1-y\hat{y}) - t/T
        """
        predicted_logits, Pts = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        b, c, t, h, w = logprobabilities.shape

        # an index at which time the classification took place
        t_index = (torch.ones(b,h,w,t) * torch.arange(t).type(torch.FloatTensor)).transpose(1,3).cuda()

        eye = torch.eye(c).type(torch.ByteTensor)
        if torch.cuda.is_available():
            eye = eye.cuda()

        # [b, t, h, w, c]
        targets_one_hot = eye[targets]

        # implement the y*\hat{y} part of the loss function
        # (permute moves classes to last dimenions -> same as targets_one_hot)
        y_haty = torch.masked_select(logprobabilities.permute(0, 2, 3, 4, 1), targets_one_hot)
        y_haty = y_haty.view(b, t, h, w).exp()

        #reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
        loss_earliness = (Pts*(t_index / t)).sum(1).mean()
        loss_classification = (Pts*(1 - y_haty)).sum(1).mean()

        loss =  alpha * loss_classification + (1 - alpha) * loss_earliness

        stats = dict(
            loss=loss,
            loss_classification=loss_classification,
            loss_earliness=loss_earliness,
        )

        return loss, logprobabilities, Pts ,stats

    def loss_cross_entropy(self, inputs, targets):

        predicted_logits, Pts = self.forward(inputs)

        logprobabilities = F.log_softmax(predicted_logits, dim=1)

        loss = F.nll_loss(logprobabilities, targets)

        stats = dict(
            loss=loss,
        )

        return loss, logprobabilities, Pts ,stats


    def predict(self, logprobabilities, Pts):
        """
        Get predicted class labels where Pts is highest
        """
        b, c, t, h, w = logprobabilities.shape
        t_class = Pts.argmax(1)  # [c x h x w]
        eye = torch.eye(t).type(torch.ByteTensor).cuda()

        prediction_all_times = logprobabilities.argmax(1)
        prediction_at_t = torch.masked_select(prediction_all_times.transpose(1, 3), eye[t_class]).view(b, h, w)
        return prediction_at_t

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
