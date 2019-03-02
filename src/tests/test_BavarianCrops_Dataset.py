import sys
sys.path.append("..")

import unittest
import os
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
class BavarianCrops_Test(unittest.TestCase):

    def test_init(self):
        region = "HOLL_2018_MT_pilot"
        root = "data/CropsDataset"

        train = BavarianCropsDataset(root=root, region=region, partition="train")
        X,y = train[0]

        self.assertTrue(os.path.exists(train.root + "/classmapping.csv"))

        self.assertEqual(train.sequencelengths.max(), 147)
        self.assertEqual(train.ndims, 13)
        #self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(train.ndims, int)
        #self.assertIsInstance(id, int)
        self.assertEqual(train.nclasses, 4)

        # check if correct files are cached
        self.assertTrue(os.path.exists(os.path.join(train.cache,"classweights.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "sequencelengths.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "y.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "ndims.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "ids.npy")))
        #self.assertTrue(os.path.exists(os.path.join(train.cache, "dataweights.npy")))

        # load again from cached files
        train.load_cached_dataset()
        self.assertIsInstance(train.ndims, int)

        # check if clean works
        train.clean_cache()
        self.assertFalse(os.path.exists(train.cache))

        valid = BavarianCropsDataset(root=root, region=region, partition="valid")


        eval = BavarianCropsDataset(root=root, region=region, partition="eval")

if __name__ == '__main__':
    unittest.main()
