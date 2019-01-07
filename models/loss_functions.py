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
    return (torch.ones(batchsize, sequencelength) * torch.arange(sequencelength).type(torch.FloatTensor)).cuda()

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


def early_loss_linear(predicted_logits, pts, targets, alpha=None, entropy_factor=0, pts_bias = 5):
    """
    Uses linear 1-P(actual class) loss. and the simple time regularization t/T
    L = (1-y\hat{y}) - t/T
    """
    logprobabilities = F.log_softmax(predicted_logits, dim=2)

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

    return loss, logprobabilities, pts, stats

def early_loss_cross_entropy(predicted_logits, pts, targets, alpha=None, entropy_factor=0, pts_bias = 5):
    logprobabilities = F.log_softmax(predicted_logits, dim=2)

    batchsize, seqquencelength, nclasses = logprobabilities.shape
    t_index = build_t_index(batchsize=batchsize,sequencelength=seqquencelength)

    # reward_earliness = (Pts * (y_haty - 1/float(self.nclasses)) * t_reward).sum(1).mean()
    loss_earliness = (1 - alpha) * ((pts+pts_bias) * (t_index / seqquencelength)).sum(1).mean()

    xentropy = F.nll_loss(logprobabilities.view(batchsize * seqquencelength, nclasses),
                                     targets.view(batchsize * seqquencelength),reduce=False).view(batchsize, seqquencelength)
    loss_classification = alpha * (pts+pts_bias)*xentropy
    loss_entropy = - entropy_factor * entropy(pts).mean()

    loss = loss_classification + loss_earliness + loss_entropy

    # NOTE adding this term even with zero factor may make results unstable -> rnn outputs nan
    if not entropy_factor == 0:
        loss = loss - entropy_factor * entropy(pts).mean()

    stats = dict(
        loss=loss,
        loss_classification=loss_classification,
        loss_earliness=loss_earliness,
    )

    return loss, logprobabilities, pts, stats

def loss_cross_entropy(predicted_logits, Pts, targets):

    logprobabilities = F.log_softmax(predicted_logits, dim=2)

    b,t,c = logprobabilities.shape
    loss = F.nll_loss(logprobabilities.view(b*t,c), targets.view(b*t))

    stats = dict(
        loss=loss,
    )

    return loss, logprobabilities, Pts ,stats
