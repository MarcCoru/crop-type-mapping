import sys
sys.path.append("..")

from utils.trainer import Trainer, CLASSIFICATION_PHASE_NAME, EARLINESS_PHASE_NAME
from datasets.Synthetic_Dataset import SyntheticDataset
from datasets.UCR_Dataset import UCRDataset
from models.DualOutputRNN import DualOutputRNN
from models.ConvShapeletModel import ConvShapeletModel
from models.ConvShapeletModel import build_n_shapelet_dict
import torch
import logging
import os

import unittest

def cleanup():
    if os.path.exists("/tmp/model_{}.pth".format(CLASSIFICATION_PHASE_NAME)):
        os.remove("/tmp/model_{}.pth".format(CLASSIFICATION_PHASE_NAME))
    if os.path.exists("/tmp/model_{}.pth".format(EARLINESS_PHASE_NAME)):
        os.remove("/tmp/model_{}.pth".format(EARLINESS_PHASE_NAME))


class TestTrainer(unittest.TestCase):

    def test_Trainer_Synthetic(self):

        try:
            traindataset = SyntheticDataset(num_samples=2000, T=100)
            validdataset = SyntheticDataset(num_samples=1000, T=100)
            nclasses = traindataset.nclasses
            traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True,
                                                          num_workers=0, pin_memory=True)

            validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=8, shuffle=False,
                                                          num_workers=0, pin_memory=True)

            model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dims=20,
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
                store="/tmp",
                overwrite=True
            )

            trainer = Trainer(model, traindataloader, validdataloader, **config)
            trainer.fit()
        except Exception as e:
            self.fail(logging.exception(e))

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

            model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dims=20,
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
                store="/tmp",
                overwrite=True,
            )

            trainer = Trainer(model, traindataloader, validdataloader, **config)
            trainer.fit()
        except Exception as e:
            self.fail(logging.exception(e))

    def test_Trainer_Conv1D_TwoPatterns(self):
        cleanup()

        try:
            traindataset = UCRDataset("TwoPatterns", partition="train", ratio=.75, randomstate=0,
                                      augment_data_noise=.1)
            validdataset = UCRDataset("TwoPatterns", partition="valid", ratio=.75, randomstate=0)
            nclasses = traindataset.nclasses
            traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True,
                                                          num_workers=0, pin_memory=True)

            validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=8, shuffle=False,
                                                          num_workers=0, pin_memory=True)

            model = ConvShapeletModel(num_layers=3,
                                      hidden_dims=50,
                                      ts_dim=1,
                                      n_classes=nclasses)

            if torch.cuda.is_available():
                model = model.cuda()

            config = dict(
                epochs=2,
                learning_rate=1e-3,
                earliness_factor=.75,
                visdomenv="unittest",
                switch_epoch=1,
                loss_mode="loss_cross_entropy",
                show_n_samples=0,
                store="/tmp"
            )

            trainer = Trainer(model, traindataloader, validdataloader, **config)
            trainer.fit()

        except Exception as e:
            self.fail(logging.exception(e))

        self.assertEqual(trainer.get_phase(), EARLINESS_PHASE_NAME)

        # should have written two model files
        self.assertTrue(os.path.exists("/tmp/model_{}.pth".format(CLASSIFICATION_PHASE_NAME)))
        self.assertTrue(os.path.exists("/tmp/model_{}.pth".format(EARLINESS_PHASE_NAME)))

    def test_build_n_shapelet_dict(self):
        n_shapelets_per_size = build_n_shapelet_dict(num_layers=3, hidden_dims=100)
        referece =  {10: 100, 20: 100, 30: 100}
        self.assertEqual(n_shapelets_per_size, referece)

        n_shapelets_per_size = build_n_shapelet_dict(num_layers=6, hidden_dims=50)
        referece =  {10: 50, 20: 50, 30: 50, 40: 50, 50: 50, 60: 50}
        self.assertEqual(n_shapelets_per_size, referece)

if __name__ == '__main__':
    unittest.main()