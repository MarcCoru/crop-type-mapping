import torch

def attentionbudget(deltas):
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
