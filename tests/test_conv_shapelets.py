import sys
sys.path.append("..")
#import os
# deactivate GPU support for tests
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import unittest
from models.conv_shapelets import ConvShapeletModel, add_time_feature_to_input
import torch


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
        logits, pts = model._logits(x=x)

        logits_sum_reference = np.array(-0.6894934, dtype=np.float32)
        self.assertAlmostEqual(logits.sum().cpu().detach().numpy(),logits_sum_reference, places=4)

        pts_sum_reference = np.array(1.1808, dtype=np.float32)
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

        logits, pts = model._logits(x=x)

        logits_sum_reference = np.array(-0.13661973, dtype=np.float32)
        self.assertAlmostEqual(logits.sum().cpu().detach().numpy(), logits_sum_reference, places=4)

        pts_sum_reference = np.array(1.1803246, dtype=np.float32)
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


if __name__ == '__main__':
    unittest.main()