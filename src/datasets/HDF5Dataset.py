import pandas as pd
import torch
import torch.utils.data
import numpy as np
import os
import sys

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, root):

        self.root = root

        self.s2_col = ['/s2/b02',
                 '/s2/b03',
                 '/s2/b04',
                 '/s2/b06',
                 '/s2/b08',
                 '/s2/b11',
                 '/s2/b12',
                 '/s2/brightness',
                 '/s2/ireci',
                 '/s2/ndvi',
                 '/s2/ndwi',
                 '/s2/scl']

        self.s1_col = ['/s1/ndvvvh', '/s1/ratiovvvh', '/s1/vh', '/s1/vv']

        if not self.cache_exists():
            self.parser = HDF5Parser(path=root)
            self.ids = self.parser.poly_ids["polyid"].unique()
            self.cache_variables()

            self.parser.close()

        self.X2, self.X1, self.y = self.load_cached_dataset()

        self.y = self.y[:,0]
        self.X = self.X2

        empty_ids = list()
        for id in np.arange(self.X2.shape[0]):
            if self.X[id].shape[0] == 0:
                empty_ids.append(id)

        self.X = np.delete(self.X,empty_ids)
        self.y = np.delete(self.y,empty_ids)

        self.ids = np.arange(self.X.shape[0])

        self.classes = np.unique(self.y)

        self.nclasses = len(self.classes)

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        self.sequencelength = self.sequencelengths.max()
        self.ndims = np.array(self.X[0]).shape[1]

        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist

        pass

    def cache_variables(self):

        base = os.path.splitext(self.root)[0]
        print("caching dataset once to "+base)

        X2s = list()
        X1s = list()
        ys = list()

        i=0
        for id in self.ids:
            if i%10==0:
                update_progress(i/float(len(self.ids)))
            i+=1

            X2,X1, y = self.load_sample(id)
            X2s.append(X2)
            X1s.append(X1)
            ys.append(y)

        X2s = np.array(X2s)
        X1s = np.array(X1s)
        ys = np.array(ys)

        np.save(base + ".S2.npy",X2s)
        np.save(base + ".S1.npy",X1s)
        np.save(base + ".y.npy",ys)

    def load_cached_dataset(self):
        X2s = np.load(os.path.splitext(self.root)[0] + ".S2.npy")
        X1s = np.load(os.path.splitext(self.root)[0] + ".S1.npy")
        ys = np.load(os.path.splitext(self.root)[0] + ".y.npy")
        return X2s, X1s, ys

    def cache_exists(self):
        base = os.path.splitext(self.root)[0]
        return os.path.exists(base + ".S2.npy") and os.path.exists(base + ".S1.npy") and os.path.exists(base + ".y.npy")


    def load_sample(self, id):
        X, y = self.parser.load_sample(polygon=id)

        X_s2 = X[self.s2_col].dropna()
        X_s1 = X[self.s1_col].dropna()

        X2 = X_s2.values
        X1 = X_s1.values
        y = np.array(y).repeat(X.shape[0])
        return X2,X1,y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        X = self.X2[idx]
        y = self.y[idx]

        y = np.array(y).repeat(X.shape[0])
        #idxs = np.random.choice(t, self.samplet, replace=False)
        #idxs.sort()
        #X = X[idxs]
        #y = y[idxs]


        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        #print(idx, X.shape, y.shape, sep=",")
        return X,y

class HDF5Parser:

    def __init__(self, path="/home/marc/data/gaf/holl_l2.h5"):

        print("opening files")
        self.h5 = pd.HDFStore(path)

        print("querying polygon ids")
        self.poly_ids = self.h5.select("/aux", columns=["polyid"]).reset_index()

        print("queryied {} rows of {} polygons".format(len(self.poly_ids), len(pd.unique(self.poly_ids["polyid"]))))

        self.features = ["/s1/ndvvvh",
                         "/s1/ratiovvvh",
                         "/s1/vh",
                         "/s1/vv",
                         "/s2/b02",
                         "/s2/b03",
                         "/s2/b04",
                         "/s2/b06",
                         "/s2/b08",
                         "/s2/b11",
                         "/s2/b12",
                         "/s2/brightness",
                         "/s2/ireci",
                         "/s2/ndvi",
                         "/s2/ndwi",
                         "/s2/scl"]

    def __len__(self):
        return len(self.poly_ids)

    def load_sample(self, polygon=1, select_method="startstop"):
        """
        selects a single polygon from the h5 dataframe

        select_method: 'where' or 'startstop'
        select by start end id is faster, but it assumes that the dataframe is sorted by polygon ID
        """

        idx = self.poly_ids.loc[self.poly_ids["polyid"] == polygon]["index"]

        if select_method == 'where':
            # select by where is slower (seemed so...)
            where = "index in (" + ", ".join([str(i) for i in idx.values]) + ")"

            bands = list()
            for feature in self.features:
                bands.append(self.h5.select(feature, where=where).median(axis=0))
        elif select_method == 'startstop':
            # select by start stop index is faster, but assumes that dataframe is sorted by polyid
            start_idx = (self.poly_ids["polyid"] == polygon).idxmax()
            end_idx = (self.poly_ids.iloc[::-1]["polyid"] == polygon).idxmax()
        else:
            raise ValueError("Invalid selectmethod. Please specify select_method either 'where' or 'startstop'")

        bands = list()
        for feature in self.features:
            bands.append(self.h5.select(feature, start=start_idx, stop=end_idx).median(axis=0))

        X = pd.concat(bands, axis=1, keys=self.features)
        y = int(self.h5.select("/aux", where="index=={}".format(polygon), columns=["classid"]).values[0])

        return X, y

    def close(self):
        self.h5.close()

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rLoaded: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

if __name__ == "__main__":

    dataset = HDF5Dataset(root="/home/marc/data/gaf/holl_l2.h5")
    X, y = dataset[0]
    #X,y = dataset[0]
    #parser.close()
