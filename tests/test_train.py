import sys
sys.path.append("..")

from train import getDataloader, getModel
import torch

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

        self.assertTrue(isinstance(traindataloader,torch.utils.data.dataloader.DataLoader))

    def test_getDataloader_synthetic(self):
        traindataloader = getDataloader(dataset="synthetic",
                                        partition="train",
                                        batch_size=32)
        self.assertTrue(len(traindataloader)==63)

    def test_getModel_DualOutputRNN(self):

        args = Namespace(
            model="DualOutputRNN",
            nclasses=2,
            hidden_dims=25,
            num_layers=2,
            dropout=0.3)

        model = getModel(args)

        self.assertTrue(len(model.state_dict())==13)

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
            nclasses=2)

        model = getModel(args)

        self.assertTrue(len(model.state_dict()) == 8)

if __name__ == '__main__':
    unittest.main()