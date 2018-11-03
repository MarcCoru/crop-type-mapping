import torch
import numpy as np
from models.DualOutputRNN import DualOutputRNN
from utils.UCR_Dataset import UCRDataset
from utils.classmetric import ClassMetric
from utils.logger import Printer

def main(batchsize=64,
    workers=4,
    epochs = 4000,
    hidden_dims = 2**8,
    learning_rate = 1e-2,
    earliness_factor=1,
    switch_epoch = 4000,
    dataset="Trace",
    savepath="tmp/model.pth",
    loadpath = "/home/marc/tmp/model_trace_e4k.pth"):

    dataset = UCRDataset(dataset, partition="eval")
    nclasses = dataset.nclasses

    # handles multitxhreaded batching and shuffling
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=workers)

    model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims)

    if torch.cuda.is_available():
        model = model.cuda()

    snapshot = model.load(path=loadpath)
    epoch = snapshot["epoch"]
    print("loaded model state at epoch " + str(epoch))

    trainargs = dict(
        switch_epoch=switch_epoch,
        earliness_factor=earliness_factor,
    )

    print()
    test_epoch(epoch, model, dataloader, trainargs)

def train_epoch(epoch, model, dataloader, optimizer, trainargs):

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

    printer.print(stats, iteration, epoch)

def test_epoch(epoch, model, dataloader, trainargs):
    printer = Printer(prefix="eval: ")

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

    printer.print(stats, iteration, epoch)

if __name__=="__main__":
    main()