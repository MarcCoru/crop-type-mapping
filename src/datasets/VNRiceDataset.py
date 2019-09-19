import torch
import torch.utils.data
import pandas as pd
import os
import sys
import numpy as np
from numpy import genfromtxt
import tqdm


BANDS = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9']
NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class VNRiceDataset(torch.utils.data.Dataset):

    def __init__(self, root, partition, mode="trainvalid", samplet=70, cache=True, seed=0, validfraction=0.2):
        assert mode in ["trainvalid", "traintest"]

        self.seed = seed
        self.validfraction = validfraction
        classmapping = os.path.join(root,"classmapping.csv")

        self.root = root

        if mode == "traintest":
            self.trainids = os.path.join(self.root, "ids", "train.txt")
            self.testids = os.path.join(self.root, "ids", "test.txt")
        elif mode == "trainvalid":
            self.trainids = os.path.join(self.root, "ids", "train.txt")
            self.testids = None

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.mapping.groupby("id").first().klassenname.values
        self.nclasses = len(self.classes)

        self.partition = partition
        self.data_folder = "{root}/csv".format(root=self.root)
        self.samplet = samplet

        #all_csv_files
        #self.csvfiles = [ for f in os.listdir(root)]
        print("Initializing VNRiceDataset {} partition".format(self.partition))

        self.cache = os.path.join(self.root,"npy",os.path.basename(classmapping), partition)

        print("read {} classes".format(self.nclasses))

        if cache and self.cache_exists() and not self.mapping_consistent_with_cache():
            self.clean_cache()

        if cache and self.cache_exists() and self.mapping_consistent_with_cache():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print("loaded {} samples".format(len(self.ids)))
        #print("class frequencies " + ", ".join(["{c}:{h}".format(h=h, c=c) for h, c in zip(self.hist, self.classes)]))

        print(self)

    def __str__(self):
        return "Dataset {}. partition {}. X:{}, y:{} with {} classes".format(self.root, self.partition,str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape, self.nclasses)

    def read_ids(self):
        assert isinstance(self.seed, int)
        assert isinstance(self.validfraction, float)
        assert self.partition in ["train", "valid", "test"]
        assert self.trainids is not None
        assert os.path.exists(self.trainids)

        np.random.seed(self.seed)

        """if trainids file provided and no testids file <- sample holdback set from trainids"""
        if self.testids is None:
            assert self.partition in ["train", "valid"]

            print("partition {} and no test ids file provided. Splitting trainids file in train and valid partitions".format(self.partition))

            with open(self.trainids,"r") as f:
                ids = [int(id) for id in f.readlines()]
            print("Found {} ids in {}".format(len(ids), self.trainids))

            np.random.shuffle(ids)

            validsize = int(len(ids) * self.validfraction)
            validids = ids[:validsize]
            trainids = ids[validsize:]

            print("splitting {} ids in {} for training and {} for validation".format(len(ids), len(trainids), len(validids)))

            assert len(validids) + len(trainids) == len(ids)

            if self.partition == "train":
                return trainids
            if self.partition == "valid":
                return validids

        elif self.testids is not None:
            assert self.partition in ["train", "test"]

            if self.partition=="test":
                with open(self.testids,"r") as f:
                    test_ids = [int(id) for id in f.readlines()]
                print("Found {} ids in {}".format(len(test_ids), self.testids))
                return test_ids

            if self.partition == "train":
                with open(self.trainids, "r") as f:
                    train_ids = [int(id) for id in f.readlines()]
                return train_ids

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition)

        ids = self.read_ids()
        assert len(ids) > 0

        self.X = list()
        self.nutzcodes = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()
        #i = 0
        for id in tqdm.tqdm(ids):

            id_file = self.data_folder+"/{id}.csv".format(id=id)
            if os.path.exists(id_file):
                self.samples.append(id_file)

                X,nutzcode = self.load(id_file)

                if len(nutzcode) > 0:
                    nutzcode = nutzcode[0]
                    if nutzcode in self.mapping.index:
                        self.X.append(X)
                        self.nutzcodes.append(nutzcode)
                        self.ids.append(id)
            else:
                self.stats["not_found"].append(id_file)

        self.y = self.applyclassmapping(self.nutzcodes)

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        assert len(self.sequencelengths) > 0
        self.sequencelength = self.sequencelengths.max()
        self.ndims = np.array(X).shape[1]

        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist
        #if 0 in self.hist:
        #    classid_ = np.argmin(self.hist)
        #    nutzid_ = self.mapping.iloc[classid_].name
        #    raise ValueError("Class {id} (nutzcode {nutzcode}) has 0 occurences in the dataset! "
        #                     "Check dataset or mapping table".format(id=classid_, nutzcode=nutzid_))


        #self.dataweights = np.array([self.classweights[y] for y in self.y])

        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.classweights)

    def mapping_consistent_with_cache(self):
        # cached y must have the same number of classes than the mapping
        return True
        #return len(np.unique(np.load(os.path.join(self.cache, "y.npy")))) == self.nclasses

    def cache_variables(self, y, sequencelengths, ids, ndims, X, classweights):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        #np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        np.save(os.path.join(self.cache, "X.npy"), X)

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"))
        self.y = np.load(os.path.join(self.cache, "y.npy"))
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"))
        self.sequencelength = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"))
        #self.dataweights = np.load(os.path.join(self.cache, "dataweights.npy"))
        self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        #dataweightsexist = os.path.exists(os.path.join(self.cache, "dataweights.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npy"))
        os.removedirs(self.cache)

    def load(self, csv_file, load_pandas = False):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        if load_pandas:
            sample = pd.read_csv(csv_file, index_col=0)
            X = np.array((sample[BANDS] * NORMALIZING_FACTOR).values)
            nutzcodes = sample["label"].values
            # nutzcode to classids (451,411) -> (0,1)

        else: # load with numpy
            data = genfromtxt(csv_file, delimiter=',', skip_header=1)
            X = data[:, 1:14] * NORMALIZING_FACTOR
            nutzcodes = data[:, 18]

        # drop times that contain nans
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0

            X = X[~t_without_nans]
            nutzcodes = nutzcodes[~t_without_nans]

        return X, nutzcodes

    def applyclassmapping(self, nutzcodes):
        """uses a mapping table to replace nutzcodes (e.g. 451, 411) with class ids"""
        return np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in nutzcodes])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        load_file = False
        if load_file:
            id = self.ids[idx]
            csvfile = os.path.join(self.data_folder, "{}.csv".format(id))
            X,nutzcodes = self.load(csvfile)
            y = self.applyclassmapping(nutzcodes=nutzcodes)
        else:

            X = self.X[idx]
            y = np.array([self.y[idx]] * X.shape[0]) # repeat y for each entry in x

        # pad up to maximum sequence length
        t = X.shape[0]

        if self.samplet is None:
            npad = self.sequencelengths.max() - t
            X = np.pad(X,[(0,npad), (0,0)],'constant', constant_values=PADDING_VALUE)
            y = np.pad(y, (0, npad), 'constant', constant_values=PADDING_VALUE)
        else:
            idxs = np.random.choice(t, self.samplet, replace=False)
            idxs.sort()
            X = X[idxs]
            y = y[idxs]


        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y, self.ids[idx]

if __name__=="__main__":
    root = "/data/vn_rice"

    region = "HOLL_2018_MT_pilot"
    classmapping = "/home/marc/data/BavarianCrops/classmapping.csv.holl"

    VNRiceDataset(root=root, partition="train", mode="trainvalid")
    VNRiceDataset(root=root, partition="valid", mode="trainvalid")

    VNRiceDataset(root=root, partition="train", mode="traintest")
    test = VNRiceDataset(root=root, partition="test", mode="traintest")

    x,y,meta = test[0]

    print(x)

