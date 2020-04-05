import sys
sys.path.append("./models")

import numpy as np
import torch
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from datasets.VNRiceDataset import VNRiceDataset
from models.TransformerEncoder import TransformerEncoder
from models.multi_scale_resnet import MSResNet
from models.TempCNN import TempCNN
from models.rnn import RNN
from datasets.ConcatDataset import ConcatDataset
from datasets.GAFDataset import GAFDataset
import argparse
from utils.trainer import Trainer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from utils.texparser import parse_run
from utils.logger import Logger
from utils.visdomLogger import VisdomLogger
from utils.scheduled_optimizer import ScheduledOptim
import torch.optim as optim
from experiments import experiments
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--batchsize', type=int, default=256, help='batch size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite automatic snapshots if they exist")
    parser.add_argument(
        '--dataroot', type=str, default='../data', help='root to dataset. default ../data')
    parser.add_argument(
        '--classmapping', type=str, default=None, help='classmapping')
    parser.add_argument(
        '--hyperparameterfolder', type=str, default=None, help='hyperparameter folder')
    parser.add_argument(
        '-x', '--experiment', type=str, default="test", help='experiment prefix')
    parser.add_argument(
        '--store', type=str, default="/tmp", help='store run logger results')
    parser.add_argument(
        '--test_every_n_epochs', type=int, default=1, help='skip some test epochs for faster overall training')
    parser.add_argument(
        '--checkpoint_every_n_epochs', type=int, default=5, help='save checkpoints during training')
    parser.add_argument(
        '--seed', type=int, default=0, help='seed for batching and weight initialization')
    parser.add_argument(
        '--hparamset', type=int, default=0, help='rank of hyperparameter set 0: best hyperparameter')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=1, help='show n samples in visdom')
    args, _ = parser.parse_known_args()

    return args

def prepare_dataset(args):

    if args.dataset == "BavarianCrops":
        root = os.path.join(args.dataroot,"BavarianCrops")

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
        root = os.path.join(args.dataroot,"GAFdataset")

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

    classmapping = args.classmapping
    hyperparameterfolder = args.hyperparameterfolder

    # prepare dataset, model, hyperparameters for the respective experiments
    args = experiments(args)

    if classmapping is not None:
        print("overwriting classmapping with manual input")
        args.classmapping = classmapping

    if hyperparameterfolder is not None:
        print("overwriting hyperparameterfolder with manual input")
        args.hyperparameterfolder = hyperparameterfolder

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
        visdomlogger=visdomlogger,
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
                              num_rnn_layers=args.num_layers, dropout=args.dropout, bidirectional=True)

    if args.model == "msresnet":
        model = MSResNet(input_channel=args.input_dims, layers=[1, 1, 1, 1], num_classes=args.nclasses, hidden_dims=args.hidden_dims)

    if args.model == "tempcnn":
        model = TempCNN(input_dim=args.input_dims, nclasses=args.nclasses, sequence_length=args.samplet, hidden_dims=args.hidden_dims, kernel_size=args.kernel_size)

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

    if torch.cuda.is_available():
        model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("initialized {} model ({} parameters)".format(args.model, pytorch_total_params))

    return model

if __name__=="__main__":

    args = parse_args()
    train(args)
