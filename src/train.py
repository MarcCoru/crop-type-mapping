import sys
sys.path.append("./models")

import numpy as np
import torch
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from datasets.VNRiceDataset import VNRiceDataset
from datasets.CropsDataset import CropsDataset
from datasets.HDF5Dataset import HDF5Dataset
from models.TransformerEncoder import TransformerEncoder
from datasets.ConcatDataset import ConcatDataset
from datasets.GAFDataset import GAFDataset
import argparse
from utils.trainer import Trainer
import os
from models.wavenet_model import WaveNetModel
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler, WeightedRandomSampler
from sampler.imbalanceddatasetsampler import ImbalancedDatasetSampler
from models.rnn import RNN
from utils.texparser import parse_run
from utils.logger import Logger
from utils.visdomLogger import VisdomLogger
from utils.scheduled_optimizer import ScheduledOptim
from models.multi_scale_resnet import MSResNet
import torch.optim as optim
from experiments import experiments
from models.TempCNN import TempCNN
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--datasets', type=str,default=None, nargs='+', help='UCR_Datasets Datasets to train on. Multiple values are allowed. '
                                                'Will also define name of the experiment. '
                                                'if not specified, will use all datasets of hyperparametercsv'
                                                '(requires --hyperparametercsv)')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '--train_on', type=str, default="train", help="dataset partition to train. Choose from 'train', 'valid', "
                                                      "'trainvalid', 'eval' (default 'train')")
    parser.add_argument(
        '--test_on', type=str, default="valid",
        help="dataset partition to train. Choose from 'train', 'valid', 'trainvalid', 'eval' (default 'valid')")
    parser.add_argument(
        '-n', '--num_layers', type=int, default=1, help='number of stacked layers. will be interpreted as stacked '
                                                        'RNN layers for recurrent models and as number of convolutions'
                                                        'for convolutional models...')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite automatic snapshots if they exist")
    parser.add_argument('--no-visdom', action='store_true',
                        help="no_visdom")
    parser.add_argument(
        '-x', '--experiment', type=str, default="test", help='experiment prefix')
    parser.add_argument(
        '--store', type=str, default="/tmp", help='store run logger results')
    parser.add_argument(
        '--test_every_n_epochs', type=int, default=1, help='skip some test epochs for faster overall training')
    parser.add_argument(
        '--checkpoint_every_n_epochs', type=int, default=5, help='save checkpoints during training')
    parser.add_argument(
        '--seed', type=int, default=None, help='seed for batching and weight initialization')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=2, help='show n samples in visdom')
    args, _ = parser.parse_known_args()

    return args

def prepare_dataset(args):

    if args.dataset == "BavarianCrops":
        root = "/data/BavarianCrops"

        #ImbalancedDatasetSampler
        test_dataset_list = list()
        for region in args.testregions:
            test_dataset_list.append(
                BavarianCropsDataset(root=root, region=region, partition=args.test_on,
                                            classmapping=args.classmapping, samplet=args.samplet,
                                     scheme=args.scheme,mode=args.mode, seed=args.seed)
            )

        train_dataset_list = list()
        for region in args.trainregions:
            train_dataset_list.append(
                BavarianCropsDataset(root=root, region=region, partition=args.train_on,
                                            classmapping=args.classmapping, samplet=args.samplet,
                                     scheme=args.scheme,mode=args.mode, seed=args.seed)
            )

    if args.dataset == "VNRice":
        train_dataset_list=[VNRiceDataset(root=args.root, partition=args.train_on, samplet=args.samplet,
                                          mode=args.mode, seed=args.seed)]

        test_dataset_list=[VNRiceDataset(root=args.root, partition=args.test_on, samplet=args.samplet,
                                         mode=args.mode, seed=args.seed)]

    if args.dataset == "BreizhCrops":
        root = "/home/marc/projects/BreizhCrops/data"

        train_dataset_list = list()
        for region in args.trainregions:
            train_dataset_list.append(
                CropsDataset(root=root, region=region, samplet=args.samplet)
            )

        #ImbalancedDatasetSampler
        test_dataset_list = list()
        for region in args.testregions:
            test_dataset_list.append(
                CropsDataset(root=root, region=region, samplet=args.samplet)
            )

    elif args.dataset == "GAFv2":
        root = "/data/GAFdataset"

        #ImbalancedDatasetSampler
        test_dataset_list = list()
        for region in args.testregions:
            test_dataset_list.append(
                GAFDataset(root, region=region, partition="test", scheme=args.scheme, classmapping=args.classmapping, features=args.features)
            )

        train_dataset_list = list()
        for region in args.trainregions:
            train_dataset_list.append(
                GAFDataset(root, region=region, partition="train", scheme=args.scheme, classmapping=args.classmapping, features=args.features)
            )

    print("setting random seed to "+str(args.seed))
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.random.manual_seed(args.seed)

    traindataset = ConcatDataset(train_dataset_list)
    traindataloader = torch.utils.data.DataLoader(dataset=traindataset, sampler=RandomSampler(traindataset),
                                                  batch_size=args.batchsize, num_workers=args.workers)

    testdataset = ConcatDataset(test_dataset_list)

    testdataloader = torch.utils.data.DataLoader(dataset=testdataset, sampler=SequentialSampler(testdataset),
                                                 batch_size=args.batchsize, num_workers=args.workers)

    return traindataloader, testdataloader

def train(args):

    # prepare dataset, model, hyperparameters for the respective experiments
    args = experiments(args)

    traindataloader, testdataloader = prepare_dataset(args)

    args.nclasses = traindataloader.dataset.nclasses
    classname = traindataloader.dataset.classname
    klassenname = traindataloader.dataset.klassenname
    args.seqlength = traindataloader.dataset.sequencelength
    #args.seqlength = args.samplet
    args.input_dims = traindataloader.dataset.ndims

    model = getModel(args)

    store = os.path.join(args.store,args.experiment)

    logger = Logger(columns=["accuracy"], modes=["train", "test"], rootpath=store)

    visdomenv = "{}_{}".format(args.experiment, args.dataset)
    visdomlogger = VisdomLogger(env=visdomenv)

    if args.model in ["transformer"]:
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay),
            model.d_model, args.warmup)
    elif args.model in ["rnn", "msresnet","tempcnn"]:
        optimizer = optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, lr=args.learning_rate)
    else:
        raise ValueError(args.model + "no valid model. either 'rnn', 'msresnet', 'transformer', 'tempcnn'")

    config = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        show_n_samples=args.show_n_samples,
        store=store,
        visdomlogger=visdomlogger if not args.no_visdom else None,
        overwrite=args.overwrite,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        test_every_n_epochs=args.test_every_n_epochs,
        logger=logger,
        optimizer=optimizer
    )

    trainer = Trainer(model,traindataloader,testdataloader,**config)
    logger = trainer.fit()

    # stores all stored values in the rootpath of the logger
    logger.save()

    #pth = store+"/npy/confusion_matrix_{epoch}.npy".format(epoch = args.epochs)
    parse_run(store, args.classmapping, outdir=store)
    #confusionmatrix2table(pth,
    #                      classnames=klassenname,
    #                      outfile=store+"/npy/table.tex")
    #texconfmat(pth)
    #accuracy2table(store+"/npy/confusion_matrix_{epoch}.npy".format(epoch = args.epochs), classnames=klassenname)



    #stats = trainer.test_epoch(evaldataloader)

    pass

def getModel(args):

    if args.model == "rnn":
        model = RNN(input_dim=args.input_dims, nclasses=args.nclasses, hidden_dims=args.hidden_dims,
                              num_rnn_layers=args.num_layers, dropout=args.dropout, bidirectional=args.bidirectional)

    if args.model == "msresnet":
        model = MSResNet(input_channel=args.input_dims, layers=[1, 1, 1, 1], num_classes=args.nclasses)

    if args.model == "tempcnn":
        model = TempCNN(input_dim=args.input_dims, nclasses=args.nclasses, sequence_length=args.samplet)

    elif args.model == "transformer":

        hidden_dims = args.hidden_dims # 256
        n_heads = args.n_heads # 8
        n_layers = args.n_layers # 6
        len_max_seq = args.samplet
        dropout = args.dropout
        d_inner = hidden_dims*4

        model = TransformerEncoder(in_channels=args.input_dims, len_max_seq=len_max_seq,
            d_word_vec=hidden_dims, d_model=hidden_dims, d_inner=d_inner,
            n_layers=n_layers, n_head=n_heads, d_k=hidden_dims//n_heads, d_v=hidden_dims//n_heads,
            dropout=dropout, nclasses=args.nclasses)

    elif args.model == "WaveNet":

        model = WaveNetModel(
                 layers=5,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=args.nclasses,
                 classes=args.nclasses,
                 output_length=1,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 input_dims=args.input_dims,
                 bias=False)


    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("initialized {} model ({} parameters)".format(args.model, pytorch_total_params))

    return model

if __name__=="__main__":

    args = parse_args()
    train(args)
