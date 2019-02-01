import sys
sys.path.append("..")
#import os
# deactivate GPU support for tests
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import unittest
from models.conv_shapelets import ConvShapeletModel, add_time_feature_to_input
import torch
import logging


class TestTrain(unittest.TestCase):
    def test_time_as_feature_equals_false(self):
        torch.manual_seed(0)
        batchsize = 2
        ts_dim = 1
        sequencelength = 5
        model = ConvShapeletModel(n_shapelets_per_size={10:50,20:50},
                                  ts_dim=ts_dim,
                                  n_classes=2,
                                  use_time_as_feature=False)
        x = torch.zeros([batchsize, ts_dim, sequencelength])
        if torch.cuda.is_available():
            x = x.cuda()
            model = model.cuda()

        logits, deltas, pts, budget = model._logits(x=x)

        logits_sum_reference = np.array(0.7290305, dtype=np.float32)
        self.assertAlmostEqual(logits.sum().cpu().detach().numpy(),logits_sum_reference, places=4)

        pts_sum_reference = np.array(2.0000002, dtype=np.float32)
        self.assertAlmostEqual(pts.sum().cpu().detach().numpy(),pts_sum_reference, places=4)

    def test_time_as_feature_equals_true(self):
        torch.manual_seed(0)
        batchsize = 2
        ts_dim = 1
        sequencelength = 5
        model = ConvShapeletModel(n_shapelets_per_size={10: 50, 20: 50},
                                  ts_dim=ts_dim,
                                  n_classes=2,
                                  use_time_as_feature=True)
        x = torch.zeros([batchsize, ts_dim, sequencelength])
        if torch.cuda.is_available():
            x = x.cuda()
            model = model.cuda()

        logits, deltas, pts, budget = model._logits(x=x)

        logits_sum_reference = np.array(-1.0345266, dtype=np.float32)
        self.assertAlmostEqual(logits.sum().cpu().detach().numpy(), logits_sum_reference, places=4)

        pts_sum_reference = np.array(2., dtype=np.float32)
        self.assertAlmostEqual(pts.sum().cpu().detach().numpy(), pts_sum_reference, places=4)

    def test_add_time_feature_to_input(self):
        batchsize = 5
        ts_dim=1
        sequencelength=10
        x = torch.zeros(batchsize,ts_dim,sequencelength)
        if torch.cuda.is_available():
            x = x.cuda()

        x_expanded = add_time_feature_to_input(x)

        self.assertEqual(list(x_expanded.shape),[batchsize,ts_dim+1,sequencelength])
        self.assertEqual(x_expanded.sum().cpu().detach().numpy(), 22.5)

    def test_defaults_constructor_arguments(self):
        try:
            ConvShapeletModel()
        except Exception as e:
            self.fail(logging.error(e))

    def test_save_load(self):

        model = ConvShapeletModel()
        try:
            model.save("/tmp/model.pth",testarg=1,secondtestarg=dict(a=1,b="c"))
            snapshot = model.load("/tmp/model.pth")
        except Exception as e:
            self.fail(logging.error(e))

        # test if custom kwargs are saved and loaded as intended...
        self.assertEqual(snapshot["testarg"],1)
        self.assertEqual(snapshot["secondtestarg"]["a"],1)
        self.assertEqual(snapshot["secondtestarg"]["b"], "c")

if __name__ == '__main__':
    unittest.main()