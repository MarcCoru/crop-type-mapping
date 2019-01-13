import torch.nn.functional as F
import torch

def entropy(p):
    return -(p*torch.log(p+1e-12)).sum(1)

def build_t_index(batchsize, sequencelength):
    # linear increasing t index for time regularization
    """
    t_index
                          0 -> T
    tensor([[ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
    batch   [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            ...,
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.],
            [ 0.,  1.,  2.,  ..., 97., 98., 99.]])
    """
    t_index = torch.ones(batchsize, sequencelength) * torch.arange(sequencelength).type(torch.FloatTensor)
    if torch.cuda.is_available():
        return t_index.cuda()
    else:
        return t_index

def build_yhaty(logprobabilities, targets):
    batchsize, seqquencelength, nclasses = logprobabilities.shape

    eye = torch.eye(nclasses).type(torch.ByteTensor)
    if torch.cuda.is_available():
        eye = eye.cuda()

    # [b, t, c]
    targets_one_hot = eye[targets]

    # implement the y*\hat{y} part of the loss function
    y_haty = torch.masked_select(logprobabilities, targets_one_hot)
    return y_haty.view(batchsize, seqquencelength).exp()


def early_loss_linear(logprobabilities, pts, targets, alpha=None, entropy_factor=0, pts_bias = 5):
    """
    Uses linear 1-P(actual class) loss. and the simple time regularization t/T
    L = (1-y\hat{y}) - t/T
    """
    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize,sequencelength=seqquencelength)

    pts_bias = pts_bias/seqquencelength

    y_haty = build_yhaty(logprobabilities, targets)

    loss_classification = alpha * ((pts+pts_bias) * (1 - y_haty)).sum(1).mean()
    loss_earliness = (1 - alpha) * ((pts+pts_bias) * (t_index / seqquencelength)).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()

    loss = loss_classification + loss_earliness + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_earliness=loss_earliness,
        loss_entropy=loss_entropy
    )

    return loss, stats

def early_loss_cross_entropy(logprobabilities, pts, targets, alpha=None, entropy_factor=0, pts_bias = 0):

    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize,sequencelength=seqquencelength)

    pts_bias = pts_bias/seqquencelength

    # reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
    loss_earliness = (1 - alpha) * ((pts+pts_bias) * (t_index / seqquencelength)).sum(1).mean()

    xentropy = F.nll_loss(logprobabilities.transpose(1,2).unsqueeze(-1), targets.unsqueeze(-1),reduction='none').squeeze(-1)
    loss_classification = alpha * ((pts+pts_bias)*xentropy).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()

    loss = loss_classification + loss_earliness + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_earliness=loss_earliness,
    )

    return loss, stats

def loss_cross_entropy(logprobabilities, pts, targets):

    b,t,c = logprobabilities.shape
    loss = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))

    stats = dict(
        loss=loss,
    )

    return loss, stats


def loss_cross_entropy_entropy_regularized(logprobabilities,pts, targets, entropy_factor=0.1):

    b,t,c = logprobabilities.shape
    #loss_classification = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))
    xentropy = F.nll_loss(logprobabilities.transpose(1, 2).unsqueeze(-1), targets.unsqueeze(-1),
                          reduction='none').squeeze(-1)
    loss_classification = (pts * xentropy).sum(1).mean()
    loss_entropy = - entropy_factor * entropy(pts).mean()
    loss = loss_classification + loss_entropy

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_entropy=loss_entropy
    )

    return loss, stats