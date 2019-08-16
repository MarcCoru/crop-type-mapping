import re
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.lines import Line2D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set_style("white")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A",
         "BRIGHTNESS", "IRECI", "NDVI", "NDWI", "NDVVVH", "RATIOVVVH", "VH", "VV"]

AGGREGATION_METHODS = ["mean", "median", "std", "p05", "p95"]

def load_raw_dataset(path='./test_train.h5'):
    testset, trainset = load_dataset(path=path)
    categories = split_column_names_into_categories(np.array(testset.columns))
    return trainset, testset, categories

def save_cache(path='./test_train.h5', cache="/tmp"):
    trainset, testset, categories = load_raw_dataset(path)

    Xtrain, Xtest, ytrain, ytest = stack(trainset, testset, categories)

    print("saving data to "+cache)
    np.save(os.path.join(cache,"Xtrain.npy"), Xtrain)
    np.save(os.path.join(cache,"Xtest.npy"), Xtest)
    np.save(os.path.join(cache,"ytrain.npy"), ytrain)
    np.save(os.path.join(cache,"ytest.npy"), ytest)

def load_cache(cache="/tmp"):
    Xtrain = np.load(os.path.join(cache, "Xtrain.npy"))
    ytrain = np.load(os.path.join(cache, "ytrain.npy"))
    Xtest = np.load(os.path.join(cache, "Xtest.npy"))
    ytest = np.load(os.path.join(cache, "ytest.npy"))
    return Xtrain, ytrain, Xtest, ytest

def main():

    save_cache(path='./test_train.h5')
    Xtrain, ytrain, Xtest, ytest = load_cache()


    plotfig, legendfig = plot()

    plotfig.savefig("/tmp/plot.png", dpi=300)
    legendfig.savefig("/tmp/legend.png", dpi=300)

    plt.show()

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

    Xtrain = np.dstack(data["Xtrain"])
    Xtest = np.dstack(data["Xtest"])

    return Xtrain, Xtest, ytrain, ytest

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
    # cache datasets -> faster loading
    if not os.path.exists("/tmp/testdataset.csv") or not os.path.exists("/tmp/traindataset.csv"):
        hdf = pd.HDFStore(path, 'r')
        testset = hdf['test_data']
        trainset = hdf['train_data']
        hdf.close()

        testset.to_csv("/tmp/testdataset.csv")
        trainset.to_csv("/tmp/traindataset.csv")
    else:
        testset = pd.read_csv("/tmp/testdataset.csv")
        trainset = pd.read_csv("/tmp/traindataset.csv")

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

def plot():
    trainset, testset, categories = load_raw_dataset(path)

    colors = 10 * ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
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
                axs[i][0].plot(traindata.transpose(), linewidth=0.5, c=colors[j], alpha=0.01,
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
    main()