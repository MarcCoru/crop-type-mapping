import torch
from models.DualOutputRNN import DualOutputRNN
from models.AttentionRNN import AttentionRNN
from models.conv_shapelets import ConvShapeletModel
from utils.UCR_Dataset import UCRDataset
from utils.Synthetic_Dataset import SyntheticDataset
import argparse
import numpy as np
import os
from utils.trainer import Trainer

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
        '--train_on', type=str, default="train", help="dataset partition to train. Choose from 'train', 'valid', 'trainvalid', 'eval' (default 'train')")
    parser.add_argument(
        '--test_on', type=str, default="valid",
        help="dataset partition to train. Choose from 'train', 'valid', 'trainvalid', 'eval' (default 'valid')")
    parser.add_argument(
        '--dropout', type=float, default=.2, help='dropout probability of the rnn layer')
    parser.add_argument(
        '-n', '--num_layers', type=int, default=1, help='number of stacked layers. will be interpreted as stacked '
                                                        'RNN layers for recurrent models and as number of convolutions'
                                                        'for convolutional models...')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of hidden dimensions per layer stacked hidden dimensions')
    parser.add_argument(
        '--augment_data_noise', type=float, default=0., help='augmentation data noise factor. defaults to 0.')
    parser.add_argument(
        '-a','--earliness_factor', type=float, default=1, help='earliness factor')
    parser.add_argument(
        '-x', '--experiment', type=str, default="test", help='experiment prefix')
    parser.add_argument(
        '--store', type=str, default="/tmp", help='store run logger results')
    parser.add_argument(
        '--run', type=str, default=None, help='run name')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=2, help='show n samples in visdom')
    parser.add_argument(
        '--loss_mode', type=str, default="twophase_early_simple", help='which loss function to choose. '
                                                                       'valid options are early_reward,  '
                                                                       'twophase_early_reward, '
                                                                       'twophase_linear_loss, or twophase_early_simple')
    parser.add_argument(
        '-s', '--switch_epoch', type=int, default=None, help='epoch at which to switch the loss function '
                                                             'from classification training to early training')

    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args


def main(args):

    traindataloader = getDataloader(dataset=args.dataset,
                                    partition=args.train_on,
                                    batch_size=args.batchsize,
                                    num_workers=args.workers,
                                    shuffle=True,
                                    pin_memory=True)

    testdataloader = getDataloader(dataset=args.dataset,
                                   partition=args.test_on,
                                   batch_size=args.batchsize,
                                   num_workers=args.workers,
                                   shuffle=False,
                                   pin_memory=True)

    args.nclasses = traindataloader.dataset.nclasses
    model = getModel(args)

    if args.run is None:
        visdomenv = "{}_{}_{}".format(args.experiment, args.dataset,args.loss_mode.replace("_","-"))
        storepath = args.store
    else:
        visdomenv = args.run
        storepath = os.path.join(args.store, args.run)

    config = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        earliness_factor=args.earliness_factor,
        visdomenv=visdomenv,
        switch_epoch=args.switch_epoch,
        loss_mode=args.loss_mode,
        show_n_samples=args.show_n_samples,
        store=storepath
    )

    trainer = Trainer(model,traindataloader,testdataloader,config=config)
    trainer.fit()

def getDataloader(dataset, partition, **kwargs):

    if dataset == "synthetic":
        torchdataset = SyntheticDataset(num_samples=2000, T=100)
    else:
        torchdataset = UCRDataset(dataset, partition=partition, ratio=.75, randomstate=0)

    # create a random seed from the partition name -> seed must be different for each partition
    seed = sum([ord(ch) for ch in partition])
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    return torch.utils.data.DataLoader(torchdataset, **kwargs)

def getModel(args):
    # Get Model
    if args.model == "DualOutputRNN":
        model = DualOutputRNN(input_dim=1, nclasses=args.nclasses, hidden_dims=args.hidden_dims,
                              num_rnn_layers=args.num_layers, dropout=args.dropout)
    elif args.model == "AttentionRNN":
        model = AttentionRNN(input_dim=1, nclasses=args.nclasses, hidden_dims=args.hidden_dims, num_rnn_layers=args.num_layers,
                             dropout=args.dropout)
    elif args.model == "Conv1D":
        model = ConvShapeletModel(num_layers=args.num_layers,
                                  hidden_dims=args.hidden_dims,
                                  ts_dim=1,
                                  n_classes=args.nclasses)
    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN', 'AttentionRNN', or 'Conv1D'")

    if torch.cuda.is_available():
        model = model.cuda()

    return model

if __name__=="__main__":

    args = parse_args()
    main(args)
