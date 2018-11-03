import torch
import numpy as np
from models.DualOutputRNN import DualOutputRNN
from utils.UCR_Dataset import UCRDataset
from utils.classmetric import ClassMetric
from utils.logger import Printer

def main(config, reporter=None):

    batchsize = config["batchsize"]
    workers = config["workers"]
    epochs = config["epochs"]
    hidden_dims = config["hidden_dims"]
    learning_rate = config["learning_rate"]
    earliness_factor = config["earliness_factor"]
    switch_epoch = config["switch_epoch"]
    num_rnn_layers = config["num_rnn_layers"]
    dataset = config["dataset"]
    savepath = config["savepath"]
    loadpath = config["loadpath"]
    silent = config["silent"]

    traindataset = UCRDataset(dataset, partition="trainvalid", ratio=.75, randomstate=2, silent=silent)
    validdataset = UCRDataset(dataset, partition="test", ratio=.75, randomstate=2, silent=silent)
    nclasses = traindataset.nclasses

    # handles multitxhreaded batching and shuffling
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers, pin_memory=True)
    validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=batchsize, shuffle=False, num_workers=workers, pin_memory=True)

    model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims, num_rnn_layers = num_rnn_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if torch.cuda.is_available():
        model = model.cuda()

    epoch=0
    if loadpath is not None:
        snapshot = model.load(path=loadpath)
        epoch = snapshot["epoch"]
        print("loaded model state at epoch " + str(epoch))
    try:
        for epoch in range(epoch,epochs):

            trainargs = dict(
                switch_epoch=switch_epoch,
                earliness_factor=earliness_factor,
            )

            if not silent:
                print()
            train_accuracy = train_epoch(epoch, model, traindataloader, optimizer, trainargs, silent)
            valid_accuracy = test_epoch(epoch, model, validdataloader, trainargs, silent)

            if reporter is not None:
                reporter(valid_accuracy=valid_accuracy)
                #reporter(train_accuracy=train_accuracy)

    finally:
        model.save(path=savepath, epoch=epoch)

def train_epoch(epoch, model, dataloader, optimizer, trainargs, silent=False):

    printer = Printer(prefix="train: ")

    # builds a confusion matrix
    metric = ClassMetric(num_classes=dataloader.dataset.nclasses)

    logged_loss_early = list()
    logged_loss_class = list()
    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        inputs, targets = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        if epoch < trainargs["switch_epoch"]:
            loss, logprobabilities = model.loss(inputs, targets)
            logged_loss_class.append(loss.detach().cpu().numpy())
        else:
            loss, logprobabilities = model.loss(inputs, targets, earliness_factor=trainargs["earliness_factor"])
            logged_loss_early.append(loss.detach().cpu().numpy())

        maxclass = logprobabilities.argmax(1)
        prediction = maxclass.mode(1)[0]

        stats = metric(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

    stats["loss_early"] = np.array(logged_loss_early).mean()
    stats["loss_class"] = np.array(logged_loss_class).mean()

    if not silent:
        printer.print(stats, iteration, epoch)

    return stats["accuracy"]

def test_epoch(epoch, model, dataloader, trainargs, silent=False):
    printer = Printer(prefix="valid: ")

    # builds a confusion matrix
    metric = ClassMetric(num_classes=dataloader.dataset.nclasses)

    logged_loss_early = list()
    logged_loss_class = list()
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            inputs, targets = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            if epoch < trainargs["switch_epoch"]:
                loss, logprobabilities = model.loss(inputs, targets)
                logged_loss_class.append(loss.detach().cpu().numpy())
            else:
                loss, logprobabilities = model.loss(inputs, targets, earliness_factor=trainargs["earliness_factor"])
                logged_loss_early.append(loss.detach().cpu().numpy())

            maxclass = logprobabilities.argmax(1)
            prediction = maxclass.mode(1)[0]

            stats = metric(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

    stats["loss_early"] = np.array(logged_loss_early).mean()
    stats["loss_class"] = np.array(logged_loss_class).mean()

    if not silent:
        printer.print(stats, iteration, epoch)

    return stats["accuracy"]

if __name__=="__main__":

    config = dict(
         batchsize=32,
         workers=4,
         epochs = 4000,
         hidden_dims = 2**10,
         learning_rate = 1e-3,
         earliness_factor=1,
         switch_epoch = 4000,
         num_rnn_layers=3,
         dataset="Trace",
         savepath="/home/marc/tmp/model_r1024_e4k.pth",
         loadpath = None,
         silent = False)

    config = dict(
        batchsize=32,
        workers=0,
        epochs=50,
        hidden_dims=2**4,
        learning_rate=1e-2,
        earliness_factor=1,
        switch_epoch=50,
        num_rnn_layers=1,
        dataset="Trace",
        savepath="/home/marc/tmp/model_r1024_e4k.pth",
        loadpath=None,
        silent=False)

    main(config)