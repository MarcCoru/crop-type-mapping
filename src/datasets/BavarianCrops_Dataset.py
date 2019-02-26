import torch
import torch.utils.data
import pandas as pd
import os
import sys
import numpy as np

BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
NORMALIZING_FACTOR = 1e-4
PADDING_VALUE = -1

class BavarianCropsDataset(torch.utils.data.Dataset):

    def __init__(self, root, region=None, partition="train", nsamples=None):
        """

        :param root:
        :param region: csv/<region>/<id>.csv
        :param partition: one of train/valid/eval
        :param nsamples: load less samples for debug
        """

        #all_csv_files
        #self.csvfiles = [ for f in os.listdir(root)]
        print("Initializing BavarianCropsDataset {} partition in region {}".format(partition, region))

        ids_file = os.path.join(root,"ids","{region}_{partition}.txt".format(region=region.lower(), partition=partition))
        with open(ids_file,"r") as f:
            ids = [int(id) for id in f.readlines()]

        # just take the first n samples for fast loading and debugging...
        if nsamples is not None:
            ids = ids[:nsamples]

        print("Found {} ids in {}".format(len(ids),ids_file))

        self.mapping = pd.read_csv(root + "/classmapping.csv", index_col=0)
        self.mapping = self.mapping.set_index("nutzcode")
        self.classes = self.mapping["id"].unique()
        self.nclasses = len(self.classes)

        print("read {} classes".format(self.nclasses))

        self.X = list()
        self.y = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()
        i = 0
        for id in ids:
            if i%500==0:
                update_progress(i/float(len(ids)))
            i+=1

            data_folder = "{root}/csv/{region}".format(root=root,region=region)
            id_file = data_folder+"/{id}.csv".format(id=id)
            if os.path.exists(id_file):
                self.samples.append(id_file)

                X,y = self.load(id_file)

                # drop times that contain nans
                if np.isnan(X).any():
                    t_without_nans = np.isnan(X).sum(1) > 0

                    X = X[~t_without_nans]
                    y = y[~t_without_nans]

                if len(y) > 0:

                    # drop samples where nutzcode is not in mapping table
                    if y[0] in self.mapping.index:

                        # replace nutzcode with class id
                        y = np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in y])

                        self.X.append(X)
                        self.y.append(y)
                        self.ids.append(id)
            else:
                self.stats["not_found"].append(id_file)

        self.T = np.array([np.array(X).shape[0] for X in self.X])
        self.sequencelength = self.T.max()
        self.ndims = np.array(X).shape[1]

        print("loaded {}/{} samples from {}".format(len(self.X),len(ids), data_folder))

    def analyze(self):

        labels = np.array([y[0] for y in self.y])
        self.classes = np.unique(labels)
        self.nclasses = len(self.classes)

        summary = list()
        for id, cl in zip(range(len(self.classes)), self.classes):
            summary.append(dict(
                nutzcode=cl,
                id=id,
                n=np.sum(labels==cl)
            ))

        return pd.DataFrame(summary)

    def load(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        sample = pd.read_csv(csv_file, index_col=0)
        X = np.array((sample[BANDS] * NORMALIZING_FACTOR).values)
        y = sample["label"].values

        return X, y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        id = self.ids[idx]

        X = self.X[idx]
        y = self.y[idx]

        # pad up to maximum sequence length
        t = X.shape[0]
        npad = self.sequencelength - t
        X = np.pad(X,[(0,npad), (0,0)],'constant', constant_values=PADDING_VALUE)
        y = np.pad(y, (0, npad), 'constant', constant_values=PADDING_VALUE)

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y

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
