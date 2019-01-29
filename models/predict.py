import torch

@torch.no_grad()
def sample_stop_decision(delta):
    dist = torch.stack([1 - delta, delta], dim=1)
    return torch.distributions.Categorical(dist).sample().byte()

def predict(logprobabilities, deltas):
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
