import torch
from models.DualOutputRNN import DualOutputRNN
from models.AttentionRNN import AttentionRNN
from utils.UCR_Dataset import UCRDataset
from utils.Synthetic_Dataset import SyntheticDataset
import argparse
import numpy as np
import os
from utils.trainer import Trainer
import pandas as pd

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
        '--dropout', type=float, default=.2, help='dropout probability of the rnn layer')
    parser.add_argument(
        '-n', '--num_rnn_layers', type=int, default=1, help='number of RNN layers')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of RNN hidden dimensions')
    parser.add_argument(
        '-a','--earliness_factor', type=float, default=1, help='earliness factor')
    parser.add_argument(
        '-x', '--experiment', type=str, default="test", help='experiment prefix')
    parser.add_argument(
        '--store', type=str, default="/tmp", help='store run logger results')
    parser.add_argument(
        '--run', type=str, default="eval", help='run name')
    parser.add_argument(
        '--load_weights', type=str, default=None, help='load model path file')
    parser.add_argument(
        '--entropy_factor', type=str, default=0, help='regularize with entropy term on the P(t) distribution high values spread P(t)')
    parser.add_argument(
        '--hparams', type=str, default=None, help='hyperparams csv file')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=2, help='show n samples in visdom')
    parser.add_argument(
        '--loss_mode', type=str, default="twophase_early_linear", help='which loss function to choose. '
                                                                       'valid options are early_reward,  '
                                                                       'twophase_early_reward, '
                                                                       'twophase_linear_loss, or twophase_cross_entropy')
    parser.add_argument(
        '-s', '--switch_epoch', type=int, default=None, help='epoch at which to switch the loss function '
                                                             'from classification training to early training')

    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

def eval(
        dataset,
        batchsize,
        workers,
        num_rnn_layers,
        dropout,
        hidden_dims,
        store="/tmp",
        epochs=30,
        switch_epoch=30,
        learning_rate=1e-3,
        run="run",
        earliness_factor=.75,
        show_n_samples=1,
        modelname="DualOutputRNN",
        loss_mode=None,
        load_weights=None,
        entropy_factor=0
    ):

    if dataset == "synthetic":
        traindataset = SyntheticDataset(num_samples=2000, T=100)
        validdataset = SyntheticDataset(num_samples=1000, T=100)
    else:
        traindataset = UCRDataset(dataset, partition="trainvalid")
        validdataset = UCRDataset(dataset, partition="test")

    nclasses = traindataset.nclasses

    np.random.seed(0)
    torch.random.manual_seed(0)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                  num_workers=workers, pin_memory=True)

    np.random.seed(1)
    torch.random.manual_seed(1)
    validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=batchsize, shuffle=False,
                                                  num_workers=workers, pin_memory=True)
    if modelname == "DualOutputRNN":
        model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims,
                              num_rnn_layers=num_rnn_layers, dropout=dropout)
    elif modelname == "AttentionRNN":
        model = AttentionRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims, num_rnn_layers=num_rnn_layers,
                             dropout=dropout)
    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN' or 'AttentionRNN'")

    if load_weights is not None:
        model.load(load_weights)

    if torch.cuda.is_available():
        model = model.cuda()

    if run is None:
        visdomenv = "{}_{}_{}".format(args.experiment, dataset,args.loss_mode.replace("_","-"))
        storepath = store
    else:
        visdomenv = run
        storepath = os.path.join(store, run)

    if switch_epoch is None:
        switch_epoch = int(epochs/2)

    config = dict(
        epochs=epochs,
        learning_rate=learning_rate,
        earliness_factor=earliness_factor,
        visdomenv=visdomenv,
        switch_epoch=switch_epoch,
        loss_mode=loss_mode,
        show_n_samples=show_n_samples,
        store=storepath,
        entropy_factor=entropy_factor
    )

    trainer = Trainer(model,traindataloader,validdataloader,config=config)
    logged_data = trainer.fit()

    return logged_data

if __name__=="__main__":

    args = parse_args()

    if args.hparams is not None:
        # get hyperparameters from the hyperparameter file for the current dataset...
        hparams = pd.read_csv(args.hparams).set_index("dataset").loc[args.dataset]
        args.hidden_dims = int(hparams["hidden_dims"])
        args.learning_rate = hparams["learning_rate"]
        args.num_rnn_layers = int(hparams["num_rnn_layers"])
        print("hyperparameter file {} provided... overwriting arguments from dataset {} with hidden_dims={}, "
              "learning_rate={}, and num_rnn_layers={}".format(args.hparams,
                                                                    args.dataset,
                                                                    args.hidden_dims,
                                                                    args.learning_rate,
                                                                    args.num_rnn_layers))

    logged_data = eval(
        dataset = args.dataset,
        batchsize = args.batchsize,
        workers = args.workers,
        num_rnn_layers = args.num_rnn_layers,
        dropout = args.dropout,
        hidden_dims = args.hidden_dims,
        store = args.store,
        epochs = args.epochs,
        switch_epoch = args.switch_epoch,
        learning_rate = args.learning_rate,
        run = args.run,
        earliness_factor = args.earliness_factor,
        show_n_samples = args.show_n_samples,
        load_weights=args.load_weights,
        loss_mode=args.loss_mode
    )
