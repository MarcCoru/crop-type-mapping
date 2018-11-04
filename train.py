import torch
import numpy as np
from models.DualOutputRNN import DualOutputRNN
from models.AttentionRNN import AttentionRNN
from utils.UCR_Dataset import UCRDataset
from utils.classmetric import ClassMetric
from utils.logger import Printer
import argparse

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

    traindataset = UCRDataset(dataset, partition="train", ratio=.75, randomstate=2, silent=silent, augment_data_noise=config["augment_data_noise"])
    validdataset = UCRDataset(dataset, partition="valid", ratio=.75, randomstate=2, silent=silent)
    nclasses = traindataset.nclasses

    # handles multitxhreaded batching and shuffling
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers, pin_memory=True)
    validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=batchsize, shuffle=False, num_workers=workers, pin_memory=True)

    if config["model"] == "DualOutputRNN":
        model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims, num_rnn_layers = num_rnn_layers)
    elif config["model"] == "AttentionRNN":
        model = AttentionRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims, num_rnn_layers = num_rnn_layers,
                             use_batchnorm=config["use_batchnorm"])
    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN' or 'AttentionRNN'")

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--dataset', type=str, default="Trace", help='UCR Dataset. Will also name the experiment')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-m', '--model', type=str, default="DualOutputRNN", help='Model variant')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-l', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument(
        '-n', '--num_rnn_layers', type=int, default=1, help='number of RNN layers')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of RNN hidden dimensions')
    parser.add_argument(
        '--use_batchnorm', action="store_true", help='use batchnorm instead of a bias vector')
    parser.add_argument(
        '--augment_data_noise', type=float, default=0., help='augmentation data noise factor. defaults to 0.')

    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":

    args = parse_args()

    main(dict(
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        model=args.model,
        earliness_factor=1,
        switch_epoch=9999,
        num_rnn_layers=args.num_rnn_layers,
        dataset=args.dataset,
        use_batchnorm=args.use_batchnorm,
        savepath="/home/marc/tmp/model_r1024_e4k.pth",
        loadpath=None,
        silent=False,
        augment_data_noise=args.augment_data_noise
    ))

    """
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
        
        """
