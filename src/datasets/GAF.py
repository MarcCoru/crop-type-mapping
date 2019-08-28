import pandas as pd
import torch
import torch.utils.data
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import re

sns.set_style("white")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A",
         "BRIGHTNESS", "IRECI", "NDVI", "NDWI", "NDVVVH", "RATIOVVVH", "VH", "VV"]

OPTICAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"]
INDICES = ["BRIGHTNESS", "IRECI", "NDVI", "NDWI", "NDVVVH", "RATIOVVVH"]
RADAR_BANDS = ["VH", "VV"]

AGGREGATION_METHODS = ["mean", "median", "std", "p05", "p95"]

class GAFDataset(torch.utils.data.Dataset):

    def __init__(self, path, region, partition, classmapping, cache="/tmp", overwrite_cache=True, features="all"):
        assert region in ["holl","nowa","krum"]


        self.hdf5_path = os.path.join(path,"test_train_{}.h5".format(region))
        self.region = region
        self.partition = partition
        self.cache = os.path.join(cache,region)

        if not self.cache_exists() or overwrite_cache:
            self.save_cache()

        if partition == "train":
            self.X = np.load(os.path.join(self.cache, "Xtrain.npy"))
            self.y = np.load(os.path.join(self.cache, "ytrain.npy"))
            self.meta = np.load(os.path.join(self.cache, "trainmeta.npy"),allow_pickle=True)
        elif partition == "test":
            self.X = np.load(os.path.join(self.cache, "Xtest.npy"))
            self.y = np.load(os.path.join(self.cache, "ytest.npy"))
            self.meta = np.load(os.path.join(self.cache, "testmeta.npy"),allow_pickle=True)
        else:
            raise ValueError("wrong partition, either 'train' or 'test'")

        # normalize optical bands
        self.X[:,:,:14] *= 1e-4
        self.X[:, :, 15] *= 1e-3
        self.X[:, :, 17] *= 1e-2

        assert features in ["all", "optical", "radar"]
        if features=="optical":
            mask = np.isin(BANDS, OPTICAL_BANDS)
            print("features='optical': selecting only {} optical features from all {} features".format(len(OPTICAL_BANDS),len(BANDS)))
            self.X = self.X[:, :, mask]

        if features=="radar":
            mask = np.isin(BANDS, RADAR_BANDS)
            print("features='radar': selecting only {} optical features from all {} features".format(len(RADAR_BANDS),
                                                                                                     len(BANDS)))
            self.X = self.X[:,:,mask]

        if region in ["holl","nowa","krum"]:
            ids_file = os.path.join(path, "{}_ids.txt".format(region))
            with open(ids_file,'r') as f:
                region_ids = [int(id.rstrip("\n")) for id in f.readlines()]

            gafids = self.meta[:,2].astype(int)

            mask = np.isin(gafids, region_ids)

            self.X = self.X[mask]
            self.y = self.y[mask]
            self.meta = self.meta[mask]

        #self.mean = self.X.mean(0).mean(0)
        #self.std = self.X.std(0).std(0)

        self.classes = np.unique(self.y)

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("nutzcode")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.mapping.groupby("id").first().klassenname.values

        self.nclasses = len(self.classes)
        self.N, self.sequencelength, self.ndims = self.X.shape
        self.sequencelengths = None

        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist


        for y in np.unique(self.y):
            try:
                self.mapping.loc[self.mapping.gafcode == y].id.iloc[0]
            except:
                pass


        print(self)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        X = self.X[idx]
        y = self.y[idx]

        y = self.mapping.loc[self.mapping.gafcode == y].id.iloc[0]


        y = np.repeat(y,self.sequencelength)
        #X -= self.mean
        #X /= self.std

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)

        return X, y, int(self.meta[idx][2])

    def applyclassmapping(self, nutzcodes):
        """uses a mapping table to replace nutzcodes (e.g. 451, 411) with class ids"""
        return np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in nutzcodes])

    def save_cache(self):
        print("saving npy arrays to " + self.cache)

        os.makedirs(self.cache, exist_ok=True)

        trainset, testset, categories = load_raw_dataset(self.hdf5_path)
        self.trainset, self.testset, self.categories = trainset, testset, categories

        Xtrain, Xtest, ytrain, ytest, testmeta, trainmeta = stack(trainset, testset, categories)

        print("saving data to " + self.cache)
        np.save(os.path.join(self.cache, "Xtrain.npy"), Xtrain)
        np.save(os.path.join(self.cache, "Xtest.npy"), Xtest)
        np.save(os.path.join(self.cache, "ytrain.npy"), ytrain)
        np.save(os.path.join(self.cache, "ytest.npy"), ytest)
        np.save(os.path.join(self.cache, "testmeta.npy"), testmeta, allow_pickle=True)
        np.save(os.path.join(self.cache, "trainmeta.npy"), trainmeta, allow_pickle=True)
        np.savetxt(os.path.join(self.cache, "trainids.csv"), trainmeta[:, 2].astype(int), fmt="%d")
        np.savetxt(os.path.join(self.cache, "testids.csv"), testmeta[:, 2].astype(int), fmt="%d")

    def cache_exists(self):
        a = os.path.exists(os.path.join(self.cache, "Xtrain.npy"))
        b = os.path.exists(os.path.join(self.cache, "Xtest.npy"))
        c = os.path.exists(os.path.join(self.cache, "ytrain.npy"))
        d = os.path.exists(os.path.join(self.cache, "ytest.npy"))
        e = os.path.exists(os.path.join(self.cache, "testmeta.npy"))
        f = os.path.exists(os.path.join(self.cache, "trainmeta.npy"))
        g = os.path.exists(os.path.join(self.cache, "trainids.csv"))
        h = os.path.exists(os.path.join(self.cache, "testids.csv"))
        return a and b and c and d and e and f and g and h

    def __str__(self):
        return "Dataset {}. region {}. partition {}. X:{}, y:{} with {} classes".format(self.hdf5_path, self.region, self.partition,self.X.shape, self.y.shape, self.nclasses)

def load_raw_dataset(path='./test_train.h5'):
    testset, trainset = load_dataset(path=path)
    categories = split_column_names_into_categories(np.array(testset.columns))
    return trainset, testset, categories

def stack(trainset, testset, categories):
    data = dict(
        Xtrain=list(),
        Xtest=list(),
    )
    for i in range(len(BANDS)):
        Xtrain, ytrain, Xtest, ytest = get_data(trainset, testset, BANDS[i], categories, "raw")
        data["Xtrain"].append(Xtrain.values)
        data["Xtest"].append(Xtest.values)

    ytest = ytest.values
    ytrain = ytrain.values

    testmeta = testset[['CRPGRPSTM', 'STMNAME', 'ID', 'coordx_lon', 'coordy_lat']].values
    trainmeta = trainset[['CRPGRPSTM', 'STMNAME', 'ID', 'coordx_lon', 'coordy_lat']].values

    Xtrain = np.dstack(data["Xtrain"])
    Xtest = np.dstack(data["Xtest"])

    return Xtrain, Xtest, ytrain, ytest, testmeta, trainmeta

def get_data(trainset, testset, band, categories, type="raw"):

    def colname2datetime(col):
        return datetime.datetime.strptime(col.split("_")[1], "%Y-%m-%d")

    cols = categories[band][type]

    dates = [colname2datetime(col) for col in cols]

    Xtrain = trainset[list(categories[band][type])]
    Xtrain.columns = dates
    ytrain = trainset["CRPGRPSTM"]
    Xtest = testset[list(categories[band][type])]
    Xtest.columns = dates
    ytest = testset["CRPGRPSTM"]

    return Xtrain, ytrain, Xtest, ytest

def load_dataset(path='./test_train.h5'):

    hdf = pd.HDFStore(path, 'r')
    testset = hdf['test_data']
    trainset = hdf['train_data']
    hdf.close()

    return testset, trainset

def split_column_names_into_categories(cols):

    """regex generator functions"""

    def get_raw_pattern(band="B02"):
        return ".*/" + band + "_[0-9]{4}-[0-9]{2}-[0-9]{2}_median"

    def get_three_month_aggregate_pattern(band="B02", aggr="median"):
        return ".*/" + band + "_median_(Jan|Feb|Mar|Apr|Mai|Jun|Jul|Aug|Sep|Oct|Nov|Dec){2}_" + aggr

    def get_annual_pattern(band="B02", aggr="median"):
        return ".*/" + band + "_median_annual_" + aggr

    categories = dict()
    for band in BANDS:
        categories[band] = dict()
        r = re.compile(get_raw_pattern(band=band))
        vmatch = np.vectorize(lambda x: bool(r.match(x)))
        idx = vmatch(cols)

        categories[band]["raw"] = cols[idx]

        for aggr in AGGREGATION_METHODS:
            r = re.compile(get_three_month_aggregate_pattern(band=band, aggr=aggr))
            vmatch = np.vectorize(lambda x: bool(r.match(x)))
            idx = vmatch(cols)
            categories[band]["3m"] = cols[idx]

            r = re.compile(get_annual_pattern(band=band, aggr=aggr))
            vmatch = np.vectorize(lambda x: bool(r.match(x)))
            idx = vmatch(cols)
            categories[band]["a"] = cols[idx]

    return categories

def plot(trainset, testset, categories):

    colors = 10 * ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
                   '#cab2d6',
                   '#6a3d9a', '#ffff99', '#b15928']

    classes = np.unique(testset["CRPGRPSTM"].unique() + trainset["CRPGRPSTM"].unique())

    plotfig, axs = plt.subplots(len(BANDS), 2, figsize=(16, 25))
    for i in range(len(BANDS)):
        Xtrain, ytrain, Xtest, ytest = get_data(trainset, testset, BANDS[i], categories, "raw")

        axs[0][0].set_title("Trainset by bands")
        axs[0][1].set_title("Testset by bands")

        axs[i][0].set_ylabel(BANDS[i])
        axs[i][1].set_ylabel(BANDS[i])
        for j in range(len(classes)):
            traindata = Xtrain.loc[ytrain == classes[j]]
            if len(traindata) > 0:
                axs[i][0].plot(traindata.transpose(), linewidth=0.5, c=colors[j], alpha=0.05,
                               zorder=np.random.randint(100))

            testdata = Xtest.loc[ytest == classes[j]]
            if len(testdata) > 0:
                axs[i][1].plot(testdata.transpose(), linewidth=0.5, c=colors[j], alpha=0.2,
                               zorder=np.random.randint(100))

    sns.despine(offset=6, left=True)

    # draw legend in a separate plot
    legendfig, ax = plt.subplots(1, 1, figsize=(16, 4))
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=classes[i]) for i in range(len(classes))]
    ax.legend(handles=legend_elements, ncol=14, loc="center")
    ax.axis("off")

    return plotfig, legendfig

if __name__=="__main__":

    ds = GAFDataset("/data/gaf/data/", partition="train", region="holl", classmapping="/data/BavarianCrops/classmapping.csv.gaf.v2")

    ds[0]

    trainset, testset, categories = load_raw_dataset(path='/home/marc/projects/crop-type-mapping/data/gaf_hdf5/test_train.h5')
    plotfig, legendfig = plot(trainset, testset, categories)
    plotfig.savefig("/tmp/plot2.png", dpi=300)
    legendfig.savefig("/tmp/legend2.png", dpi=300)

    trainset, testset, categories = ds.trainset, ds.testset, ds.categories
    plotfig, legendfig = plot(trainset, testset, categories)
    plotfig.savefig("/tmp/plot3.png", dpi=300)
    legendfig.savefig("/tmp/legend3.png", dpi=300)

    plt.show()