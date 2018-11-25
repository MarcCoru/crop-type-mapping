import sys
sys.path.append("..")

from utils.trainer import Trainer
from utils.Synthetic_Dataset import SyntheticDataset
from utils.UCR_Dataset import UCRDataset
from models.DualOutputRNN import DualOutputRNN
from models.AttentionRNN import AttentionRNN
import torch

import unittest

class TestTrain(unittest.TestCase):

    def test_Trainer_Synthetic(self):

        try:
            traindataset = SyntheticDataset(num_samples=2000, T=100)
            validdataset = SyntheticDataset(num_samples=1000, T=100)
            nclasses = traindataset.nclasses
            traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True,
                                                          num_workers=0, pin_memory=True)

            validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=8, shuffle=False,
                                                          num_workers=0, pin_memory=True)

            model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=20,
                                  num_rnn_layers=1, dropout=.2)

            if torch.cuda.is_available():
                model = model.cuda()

            config = dict(
                epochs=2,
                learning_rate=1e-3,
                earliness_factor=.75,
                visdomenv="unittest",
                switch_epoch=1,
                loss_mode="twophase_linear_loss",
                show_n_samples=0,
                store="/tmp"
            )

            trainer = Trainer(model, traindataloader, validdataloader, config=config)
            trainer.fit()
        except Exception as e:
            self.fail("Failed tests: "+str(e))

    def test_Trainer_TwoPatterns(self):

        try:
            traindataset = UCRDataset("TwoPatterns", partition="train", ratio=.75, randomstate=0,
                                      augment_data_noise=.1)
            validdataset = UCRDataset("TwoPatterns", partition="valid", ratio=.75, randomstate=0)
            nclasses = traindataset.nclasses
            traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True,
                                                          num_workers=0, pin_memory=True)

            validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=8, shuffle=False,
                                                          num_workers=0, pin_memory=True)

            model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=20,
                                  num_rnn_layers=1, dropout=.2)

            if torch.cuda.is_available():
                model = model.cuda()

            config = dict(
                epochs=2,
                learning_rate=1e-3,
                earliness_factor=.75,
                visdomenv="unittest",
                switch_epoch=1,
                loss_mode="twophase_linear_loss",
                show_n_samples=0,
                store="/tmp"
            )

            trainer = Trainer(model, traindataloader, validdataloader, config=config)
            trainer.fit()
        except Exception as e:
            self.fail("Failed tests: "+str(e))

    def test_Trainer_AttentionRNN_TwoPatterns(self):

        try:
            traindataset = UCRDataset("TwoPatterns", partition="train", ratio=.75, randomstate=0,
                                      augment_data_noise=.1)
            validdataset = UCRDataset("TwoPatterns", partition="valid", ratio=.75, randomstate=0)
            nclasses = traindataset.nclasses
            traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True,
                                                          num_workers=0, pin_memory=True)

            validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=8, shuffle=False,
                                                          num_workers=0, pin_memory=True)

            model = AttentionRNN(input_dim=1, nclasses=nclasses, hidden_dim=20,
                                 num_rnn_layers=1,
                                 dropout=.2)

            if torch.cuda.is_available():
                model = model.cuda()

            config = dict(
                epochs=2,
                learning_rate=1e-3,
                earliness_factor=.75,
                visdomenv="unittest",
                switch_epoch=1,
                loss_mode="twophase_linear_loss",
                show_n_samples=0,
                store="/tmp"
            )

            trainer = Trainer(model, traindataloader, validdataloader, config=config)
            trainer.fit()
        except Exception as e:
            self.fail("Failed tests: "+str(e))
