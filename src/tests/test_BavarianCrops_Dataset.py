import sys
sys.path.append("..")

import unittest
import os
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
import pandas as pd
import numpy as np
import shutil

class BavarianCrops_Test(unittest.TestCase):

    def test_init(self):
        region = "regionA"
        root = "data/CropsDataset"

        train = BavarianCropsDataset(root=root, region=region, partition="train")
        X,y = train[0]

        self.assertTrue(os.path.exists(train.root + "/classmapping.csv"))

        self.assertEqual(train.sequencelengths.max(), 71)
        self.assertEqual(train.ndims, 13)
        #self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIsInstance(train.ndims, int)
        #self.assertIsInstance(id, int)
        self.assertEqual(train.nclasses, 22)

        # check if correct files are cached
        self.assertTrue(os.path.exists(os.path.join(train.cache, "classweights.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "sequencelengths.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "y.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "ndims.npy")))
        self.assertTrue(os.path.exists(os.path.join(train.cache, "ids.npy")))
        #self.assertTrue(os.path.exists(os.path.join(train.cache, "dataweights.npy")))

        self.assertIsInstance(train.mapping, pd.core.frame.DataFrame)
        self.assertIsInstance(train.classes, np.ndarray)
        self.assertIsInstance(train.classname, np.ndarray)
        self.assertIsInstance(train.klassenname, np.ndarray)
        self.assertIsInstance(train.nclasses, int)

        # load again from cached files
        train.load_cached_dataset()
        self.assertIsInstance(train.ndims, int)

        # check if clean works
        train.clean_cache()
        self.assertFalse(os.path.exists(train.cache))

        valid = BavarianCropsDataset(root=root, region=region, partition="valid")

        eval = BavarianCropsDataset(root=root, region=region, partition="eval")

        # cleanup
        shutil.rmtree("data/CropsDataset/npy")

if __name__ == '__main__':
    unittest.main()
