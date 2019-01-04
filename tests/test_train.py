import sys

sys.path.append("..")

import train
from train import getDataloader, getModel, readHyperparameterCSV

import torch
import argparse
import os

import unittest


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestTrain(unittest.TestCase):
    def test_getDataloader_Trace(self):
        traindataloader = getDataloader(dataset="Trace",
                                        partition="train",
                                        batch_size=32)
        self.assertTrue(len(traindataloader) == 3)

        testdataloader = getDataloader(dataset="Trace",
                                       partition="test",
                                       batch_size=32)
        self.assertTrue(len(testdataloader) == 4)

        validdataloader = getDataloader(dataset="Trace",
                                        partition="valid",
                                        batch_size=32)
        self.assertTrue(len(validdataloader) == 1)

        trainvaliddataloader = getDataloader(dataset="Trace",
                                             partition="trainvalid",
                                             batch_size=32)
        self.assertTrue(len(trainvaliddataloader) == 4)

        self.assertTrue(isinstance(traindataloader, torch.utils.data.dataloader.DataLoader))

    def test_getDataloader_synthetic(self):
        traindataloader = getDataloader(dataset="synthetic",
                                        partition="train",
                                        batch_size=32)
        self.assertTrue(len(traindataloader) == 63)

    def test_getModel_DualOutputRNN(self):
        args = Namespace(
            model="DualOutputRNN",
            nclasses=2,
            hidden_dims=25,
            num_layers=2,
            dropout=0.3)

        model = getModel(args)

        self.assertTrue(len(model.state_dict()) == 13)

    def test_getModel_AttentionRNN(self):
        args = Namespace(
            model="AttentionRNN",
            nclasses=2,
            hidden_dims=25,
            num_layers=2,
            dropout=0.3)

        model = getModel(args)

        self.assertTrue(len(model.state_dict()) == 23)

    def test_getModel_Conv1D(self):
        args = Namespace(
            model="Conv1D",
            num_layers=2,
            hidden_dims=20,
            ts_dim=1,
            nclasses=2,
            seqlength=100)

        model = getModel(args)

        self.assertTrue(len(model.state_dict()) == 13)

    def test_readHyperparameterCSV_conv1d(self):
        args = Namespace(
            hyperparametercsv="data/hyperparams_conv1d.csv",
            dataset="Trace",
            num_layers=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            hidden_dims=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            ts_dim=1,
            nclasses=2,
            dropout=0.5)

        args = readHyperparameterCSV(args)

        # those values should have been overwritten by the CSV hyperparameter file
        self.assertTrue(args.hidden_dims == 25)
        self.assertTrue(args.num_layers == 4)

        # make sure the datatype from the csv (default float) is overwritten by the datatype from previous argument
        self.assertTrue(isinstance(args.hidden_dims, int))

        # must be instance of argparse namespace
        self.assertTrue(isinstance(args, argparse.Namespace))

    def test_readHyperparameterCSV_rnn(self):
        args = Namespace(
            hyperparametercsv="data/hyperparams_rnn.csv",
            dataset="Trace",
            num_rnn_layers=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            hidden_dims=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            ts_dim=1,
            nclasses=2)

        args = readHyperparameterCSV(args)

        # those values should have been overwritten by the CSV hyperparameter file
        self.assertTrue(args.hidden_dims == 512)
        self.assertTrue(args.num_rnn_layers == 4)

        # make sure the datatype from the csv (default float) is overwritten by the datatype from previous argument
        self.assertTrue(isinstance(args.hidden_dims, int))

        # must be instance of argparse namespace
        self.assertTrue(isinstance(args, argparse.Namespace))

    def test_train_Conv1D(self):
        args = Namespace(
            batchsize=128,
            dataset='Trace',
            earliness_factor=0.75,
            epochs=2,
            switch_epoch=1,
            experiment='test',
            hidden_dims=50,
            hyperparametercsv="data/hyperparams_conv1d.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='Conv1D',
            num_layers=3,
            show_n_samples=1,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2)

        train.main(args)

        self.assertTrue(os.path.exists("/tmp/unittest/model_1.pth"))

    def test_train_DualOutputRNN(self):
        args = Namespace(
            batchsize=128,
            dataset='Trace',
            earliness_factor=0.75,
            epochs=2,
            switch_epoch=1,
            experiment='test',
            hidden_dims=50,
            hyperparametercsv="data/hyperparams_rnn.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='DualOutputRNN',
            num_layers=3,
            show_n_samples=1,
            dropout=0.4,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2)

        train.main(args)

        self.assertTrue(os.path.exists("/tmp/unittest/model_1.pth"))

    def test_train_AttentionRNN(self):
        args = Namespace(
            batchsize=128,
            dataset='Trace',
            earliness_factor=0.75,
            epochs=2,
            switch_epoch=1,
            experiment='test',
            hidden_dims=50,
            hyperparametercsv="data/hyperparams_rnn.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='AttentionRNN',
            num_layers=3,
            show_n_samples=1,
            dropout=0.4,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2)

        train.main(args)

        self.assertTrue(os.path.exists("/tmp/unittest/model_1.pth"))


if __name__ == '__main__':
    unittest.main()
