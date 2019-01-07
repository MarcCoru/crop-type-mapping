import sys
sys.path.append("..")

import numpy as np
import unittest
from models.conv_shapelets import ConvShapeletModel, add_time_feature_to_input
import torch
import logging
from models.loss_functions import early_loss_linear
from models.conv_shapelets import ConvShapeletModel
from models.DualOutputRNN import DualOutputRNN

class TestLossFunctions(unittest.TestCase):

    def test_Conv1D(self):
        batchsize = 25
        sequencelength = 398
        ts_dim = 1

        model = ConvShapeletModel(ts_dim=ts_dim)
        inputs = torch.zeros([batchsize, sequencelength, ts_dim])
        targets = torch.zeros([batchsize, sequencelength],dtype=torch.long)

        if torch.cuda.is_available():
            model = model.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()

        out_tuple = model.early_loss_linear(inputs=inputs, targets=targets, alpha=1, entropy_factor=0.1)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)

        out_tuple = model.early_loss_cross_entropy(inputs=inputs, targets=targets, alpha=1, entropy_factor=0.1)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)

        out_tuple = model.loss_cross_entropy(inputs=inputs, targets=targets)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)


    def test_DualOutputRNN(self):
        batchsize = 25
        sequencelength = 398
        ts_dim = 1

        model = DualOutputRNN()
        inputs = torch.zeros([batchsize, sequencelength, ts_dim])
        targets = torch.zeros([batchsize, sequencelength],dtype=torch.long)

        if torch.cuda.is_available():
            model = model.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()

        out_tuple = model.early_loss_linear(inputs=inputs, targets=targets, alpha=1, entropy_factor=0.1)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)

        out_tuple = model.early_loss_cross_entropy(inputs=inputs, targets=targets, alpha=1, entropy_factor=0.1)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)

        out_tuple = model.loss_cross_entropy(inputs=inputs, targets=targets)
        loss, logprobabilities, pts, stats = out_tuple
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.cpu().detach().numpy().dtype==np.float32)



if __name__ == '__main__':
    unittest.main()