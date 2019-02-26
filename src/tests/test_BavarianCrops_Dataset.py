import sys
sys.path.append("..")

import unittest
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
class BavarianCrops_Test(unittest.TestCase):

    def test_init(self):
        region = "HOLL_2018_MT_pilot"
        root = "/data/BavarianCrops"

        train = BavarianCropsDataset(root=root, region=region, partition="train")
        X,y = train[0]

        train.analyze()

        self.assertTrue(train.sequencelength,147)
        self.assertTrue(train.ndims, 11)
        self.assertTrue(len(train), 4210)
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        #self.assertIsInstance(id, int)
        self.assertEqual(train.nclasses, 121)

        valid = BavarianCropsDataset(root=root, region=region, partition="valid")
        self.assertTrue(len(valid), 4210)

        eval = BavarianCropsDataset(root=root, region=region, partition="eval")
        self.assertTrue(len(eval), 4210)

if __name__ == '__main__':
    unittest.main()
