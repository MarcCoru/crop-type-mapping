from models.duplo import DuPLO
import torch
from train import prepare_dataset
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import sklearn.metrics
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', type=str, default="12classes", help='either 12classes or 23classes')
    parser.add_argument(
        'dataset', type=str, default="BavarianCrops", help='either BavarianCrops or GAFv2')
    args, _ = parser.parse_known_args()

    return args


def main(args):

    traindataloader, testdataloader, model, args, device = setup(args.dataset, args.mode)

    optimizer = torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, lr=args.learning_rate)

    stats = list()
    for epoch in range(args.epochs):
        trainlosses = train_epoch(traindataloader, optimizer, model, device)
        testlosses, predictions, labels = test_epoch(testdataloader, model, device)

        trainloss = np.array(trainlosses).mean()
        testloss = np.array(testlosses).mean()
        stat = metrics(labels, predictions)
        stat["epoch"] = epoch
        stat["trainloss"] = trainloss
        stat["testloss"] = testloss
        stats.append(stat)
        print(f"epoch {epoch}: trainloss: {trainloss:.2f}, testloss: {testloss:.2f}, kappa: {stat['kappa']:.2f}, accuracy: {stat['accuracy']:.2f}")

        stat_df = pd.DataFrame(stats).set_index("epoch")

        os.makedirs(os.path.join(args.store, args.experiment), exist_ok=True)
        stat_df.to_csv(os.path.join(args.store,args.experiment, "log.csv"))

        better = not (stat_df["testloss"] < stat["testloss"]).any()
        if better:
            model.save(os.path.join(args.store,args.experiment, "model.pth"))


def setup(dataset, mode, dataroot="../data", store='/tmp/'):
    if mode == "12classes":
        classmapping = os.path.join(dataroot, "BavarianCrops", 'classmapping12.csv')
    elif mode == "23classes":
        classmapping = os.path.join(dataroot, "BavarianCrops", 'classmapping23.csv')

    args = Namespace(batchsize=256,epochs=1500,
                     classmapping=classmapping,
                     dataroot=dataroot, dataset=dataset,
                     model='duplo', mode=None,weight_decay=0, learning_rate=1e-3,
                     seed=0, store=store, workers=0)

    if dataset == "BavarianCrops":
        args = merge([args, TUM_dataset])
        exp = "isprs_tum_duplo"
    elif dataset == "GAFv2":
        args = merge([args, GAF_dataset])
        exp = "isprs_gaf_duplo"
    args.experiment = exp
    args.store = f"/tmp/{mode}"

    args.train_on = "train"
    args.test_on = "valid"
    traindataloader, testdataloader = prepare_dataset(args)

    input_dim = traindataloader.dataset.datasets[0].ndims
    nclasses = len(traindataloader.dataset.datasets[0].classes)

    device = torch.device("cuda")
    model = DuPLO(input_dim=input_dim, nclasses=nclasses, sequencelength=args.samplet, dropout=0.4)

    model.to(device)

    return traindataloader, testdataloader, model, args, device


def train_epoch(dataloader, optimizer, model, device):
    losses = list()
    model.train()
    for iteration, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        optimizer.zero_grad()

        inputs, targets, _ = data

        inputs = inputs.to(device)
        targets = targets.to(device)

        logprobabilities, logprobabilities_cnn, logprobabilities_rnn = model.forward(inputs.transpose(1,2))

        loss = torch.nn.functional.nll_loss(logprobabilities, targets[:, 0])
        loss_cnn = torch.nn.functional.nll_loss(logprobabilities_cnn, targets[:, 0])
        loss_rnn = torch.nn.functional.nll_loss(logprobabilities_rnn, targets[:, 0])

        loss += 0.3 * loss_cnn
        loss += 0.3 * loss_rnn

        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        return losses

def test_epoch(dataloader, model, device):
    predictions = list()
    losses = list()
    labels = list()
    model.eval()

    with torch.no_grad():
        for iteration, data in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):

            inputs, targets, _ = data

            inputs = inputs.to(device)
            targets = targets.to(device)

            logprobabilities, logprobabilities_cnn, logprobabilities_rnn = model.forward(inputs.transpose(1, 2))

            loss = torch.nn.functional.nll_loss(logprobabilities, targets[:, 0])
            loss_cnn = torch.nn.functional.nll_loss(logprobabilities_cnn, targets[:, 0])
            loss_rnn = torch.nn.functional.nll_loss(logprobabilities_rnn, targets[:, 0])

            loss += 0.3 * loss_cnn
            loss += 0.3 * loss_rnn

            losses.append(loss.detach().cpu().numpy())
            y_pred = logprobabilities.argmax(-1)

            predictions.append(y_pred.detach().cpu().numpy())
            labels.append(targets[:, 0].detach().cpu().numpy())

    return losses, np.hstack(predictions), np.hstack(labels)

def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)

TUM_dataset = Namespace(
    dataset = "BavarianCrops",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    scheme="blocks",
    test_on = "test",
    train_on = "trainvalid",
    samplet = 70
)

GAF_dataset = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    features = "optical",
    scheme="blocks",
    test_on="test",
    train_on="train",
    samplet = 23
)

if __name__ == '__main__':
    args = parse_args()
    main(args)