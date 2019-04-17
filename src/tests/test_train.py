import sys

sys.path.append("..")
sys.path.append("../models")

import train
from train import getModel, readHyperparameterCSV, parse_dataset_names, get_datasets_from_hyperparametercsv

import torch
import argparse
import os

import unittest


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestTrain(unittest.TestCase):

    def test_getModel_RNN(self):
        args = Namespace(
            model="rnn",
            nclasses=2,
            input_dims=13,
            hidden_dims=25,
            num_layers=2,
            dropout=0.3)

        model = getModel(args)

        self.assertEquals(len(model.state_dict()),14)

    def test_getModel_Transfomer(self):
        args = Namespace(model="transformer", input_dims=13, nclasses=2)

        model = getModel(args)

        self.assertEquals(len(model.state_dict()),106)


if __name__ == '__main__':
    unittest.main()
