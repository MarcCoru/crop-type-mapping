import sys

sys.path.append("..")

import train
from train import getDataloader, getModel, readHyperparameterCSV, parse_dataset_names, get_datasets_from_hyperparametercsv

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

    def test_getModel_Conv1D(self):
        args = Namespace(
            model="Conv1D",
            num_layers=2,
            hidden_dims=20,
            ts_dim=1,
            nclasses=2,
            seqlength=100,
            shapelet_width_in_percent=False,
            dropout=0.5,
            shapelet_width_increment=10)

        model = getModel(args)

        self.assertTrue(len(model.state_dict()) == 13)

    def test_readHyperparameterCSV_conv1d(self):
        args = Namespace(
            hyperparametercsv="tests/data/hyperparams_conv1d.csv",
            dataset="Trace",
            num_layers=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            hidden_dims=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            ts_dim=1,
            nclasses=2,
            dropout=0.5,
            learning_rate = 0.1
        )

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
            hyperparametercsv="tests/data/hyperparams_rnn.csv",
            dataset="Trace",
            num_rnn_layers=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            hidden_dims=999,  # <- should be overwritten by the csv file, but datatype should be preserved
            ts_dim=1,
            nclasses=2,
            learning_rate=0.1)

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
            hyperparametercsv="tests/data/hyperparams_conv1d.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='Conv1D',
            num_layers=3,
            show_n_samples=1,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2,
            shapelet_width_in_percent=False,
            dropout=0.5,
            shapelet_width_increment=10,
            overwrite=True,
            test_every_n_epochs=1,
            train_valid_split_ratio=0.75,
            train_valid_split_seed=0,
            entropy_factor=0.1,
            resume_optimizer=False,
            epsilon=0)

        train.train(args)

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
            hyperparametercsv="tests/data/hyperparams_rnn.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='DualOutputRNN',
            num_layers=3,
            show_n_samples=1,
            dropout=0.4,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2,
            overwrite=True,
            test_every_n_epochs=1,
            train_valid_split_ratio=0.75,
            train_valid_split_seed=0,
            entropy_factor=0.1,
            resume_optimizer = False,
            epsilon=0
        )

        train.train(args)

        self.assertTrue(os.path.exists("/tmp/unittest/model_1.pth"))

    def test_train_AttentionRNN(self):
        return # TODO fix AttentionRNN

        args = Namespace(
            batchsize=128,
            dataset='Trace',
            earliness_factor=0.75,
            epochs=2,
            switch_epoch=1,
            experiment='test',
            hidden_dims=50,
            hyperparametercsv="tests/data/hyperparams_rnn.csv",
            learning_rate=0.01,
            loss_mode='twophase_linear_loss',
            model='AttentionRNN',
            num_layers=3,
            show_n_samples=1,
            dropout=0.4,
            store='/tmp',
            test_on='valid',
            train_on='train',
            workers=2,
            overwrite=True,
            test_every_n_epochs=1,
            train_valid_split_ratio=0.75,
            train_valid_split_seed=0,
            entropy_factor=0.1,
            resume_optimizer = False,
            epsilon=0
        )

        train.train(args)

        self.assertTrue(os.path.exists("/tmp/unittest/model_1.pth"))


    def test_parse_dataset_names(self):

        #,should raise error on dataset 'Someotherdataset' not being in hyperparameters
        with self.assertRaises(ValueError):
            parse_dataset_names(
                Namespace(
                    hyperparametercsv="tests/data/hyperparams_conv1d.csv",
                    datasets=["Trace","Someotherdataset","TwoPatterns"]
                )
            )

        # should raise error asking for more input
        with self.assertRaises(ValueError):
            parse_dataset_names(
                Namespace(
                    hyperparametercsv=None,
                    datasets=None
                )
            )

        args = parse_dataset_names(
                Namespace(
                    hyperparametercsv=None,
                    datasets=["Trace"]
                )
            )
        self.assertEqual(args.datasets,["Trace"])

        args = parse_dataset_names(
            Namespace(
                hyperparametercsv="tests/data/hyperparams_conv1d.csv",
                datasets=None
            )
        )
        self.assertEqual(len(args.datasets), 38, "expected 38 datasets in 'tests/data/hyperparams_conv1d.csv'")

        args = parse_dataset_names(
            Namespace(
                hyperparametercsv="tests/data/hyperparams_conv1d.csv",
                datasets=["TwoPatterns","InlineSkate"]
            )
        )
        self.assertEqual(args.datasets, ["TwoPatterns","InlineSkate"], "expected 2 datasets that are also "
                                                                       "present in 'tests/data/hyperparams_conv1d.csv'")

    def test_get_datasets_from_hyperparametercsv(self):
        datasets = get_datasets_from_hyperparametercsv("tests/data/hyperparams_conv1d.csv")

        ref_datasets = ['NonInvasiveFatalECGThorax2',  'SonyAIBORobotSurface2',  'MoteStrain',  'CricketZ',  'FacesUCR',  'NonInvasiveFatalECGThorax1',  'SwedishLeaf',  'Symbols',  'FaceAll',  'GunPoint',  'OSULeaf',  'StarLightCurves',  'DiatomSizeReduction',  'FiftyWords',  'CBF',  'WordSynonyms',  'ECGFiveDays',  'TwoLeadECG',  'CricketX',  'FaceFour',  'Mallat',  'OliveOil',  'UWaveGestureLibraryX',  'SonyAIBORobotSurface1',  'Haptics',  'Trace',  'Yoga',  'SyntheticControl',  'Fish',  'Adiac',  'ChlorineConcentration',  'CricketY',  'CinCECGTorso',  'Beef',  'InlineSkate',  'MedicalImages',  'ItalyPowerDemand',  'TwoPatterns']

        self.assertEqual(datasets,ref_datasets)
        self.assertIsInstance(datasets,list)

if __name__ == '__main__':
    unittest.main()
