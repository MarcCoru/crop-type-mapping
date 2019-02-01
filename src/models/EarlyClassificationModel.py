from abc import ABC, abstractmethod
import torch
from sklearn.base import BaseEstimator

class EarlyClassificationModel(ABC,torch.nn.Module, BaseEstimator):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass # return logprobabilities, deltas, pts, budget

    @torch.no_grad()
    def predict(self, logprobabilities, deltas):

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().byte()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(stop_now.shape).byte()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).byte()

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.argmax(1)

        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return predictions_at_t_stop, t_stop

    def attentionbudget(self,deltas):
        batchsize, sequencelength = deltas.shape

        pts = list()

        initial_budget = torch.ones(batchsize)
        if torch.cuda.is_available():
            initial_budget = initial_budget.cuda()

        budget = [initial_budget]
        for t in range(1, sequencelength):
            pt = deltas[:, t] * budget[-1]
            budget.append(budget[-1] - pt)
            pts.append(pt)

        # last time
        pt = budget[-1]
        budget.append(budget[-1] - pt)
        pts.append(pt)

        return torch.stack(pts, dim=-1), torch.stack(budget, dim=1)

    @abstractmethod
    def save(self, path="model.pth",**kwargs):
        pass

    @abstractmethod
    def load(self, path):
        pass #return snapshot