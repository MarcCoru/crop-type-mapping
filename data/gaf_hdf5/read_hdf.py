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

sns.set_style("whitegrid")

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# cache datasets -> faster loading
if not os.path.exists("/tmp/testdataset.csv") or not os.path.exists("/tmp/traindataset.csv"):
    hdf =  pd.HDFStore('./test_train.h5','r')
    testset =  hdf['test_data']
    trainset =  hdf['train_data']
    hdf.close()

    testset.to_csv("/tmp/testdataset.csv")
    trainset.to_csv("/tmp/traindataset.csv")
else:
    testset = pd.read_csv("/tmp/testdataset.csv")
    trainset = pd.read_csv("/tmp/traindataset.csv")

def get_raw_pattern(band="B02"):
    return ".*/"+band+"_[0-9]{4}-[0-9]{2}-[0-9]{2}_median"

def get_three_month_aggregate_pattern(band="B02", aggr="median"):
    return ".*/"+band+"_median_(Jan|Feb|Mar|Apr|Mai|Jun|Jul|Aug|Sep|Oct|Nov|Dec){2}_"+aggr

def get_annual_pattern(band="B02", aggr="median"):
    return ".*/" + band + "_median_annual_"+aggr

bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A",
         "BRIGHTNESS", "IRECI", "NDVI", "NDWI", "NDVVVH", "RATIOVVVH", "VH", "VV"]

aggregation_methods = ["mean", "median", "std", "p05", "p95"]

cols = np.array(testset.columns)

categories = dict()
for band in bands:
    categories[band] = dict()
    r = re.compile(get_raw_pattern(band=band))
    vmatch = np.vectorize(lambda x: bool(r.match(x)))
    idx = vmatch(cols)

    categories[band]["raw"] = cols[idx]

    for aggr in aggregation_methods:

        r = re.compile(get_three_month_aggregate_pattern(band=band, aggr=aggr))
        vmatch = np.vectorize(lambda x: bool(r.match(x)))
        idx = vmatch(cols)
        categories[band]["3m"] = cols[idx]

        r = re.compile(get_annual_pattern(band=band, aggr=aggr))
        vmatch = np.vectorize(lambda x: bool(r.match(x)))
        idx = vmatch(cols)
        categories[band]["a"] = cols[idx]

def colname2datetime(col):
    return datetime.datetime.strptime(col.split("_")[1], "%Y-%m-%d")

def get_data(band, type="raw"):
    cols = categories[band][type]

    dates = [colname2datetime(col) for col in cols]

    Xtrain = trainset[list(categories[band][type])]
    Xtrain.columns = dates
    ytrain = trainset["CRPGRPSTM"]
    Xtest = testset[list(categories[band][type])]
    Xtest.columns = dates
    ytest = testset["CRPGRPSTM"]

    return Xtrain, ytrain, Xtest, ytest

colors = 10 * ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

classes = np.unique(testset["CRPGRPSTM"].unique() + trainset["CRPGRPSTM"].unique())


fig, axs = plt.subplots(len(bands)+1,2, figsize=(16,25))
for i in range(len(bands)):
    Xtrain, ytrain, Xtest, ytest = get_data(bands[i], "raw")

    axs[0][0].set_title("Trainset by bands")
    axs[0][1].set_title("Testset by bands")

    axs[i][0].set_ylabel(bands[i])
    axs[i][1].set_ylabel(bands[i])
    for j in range(len(classes)):
        traindata = Xtrain.loc[ytrain == classes[j]]
        if len(traindata) > 0:
            axs[i][0].plot(traindata.transpose(), linewidth=0.5, c=colors[j], alpha=0.05, zorder=np.random.randint(100))

        testdata = Xtest.loc[ytest == classes[j]]
        if len(testdata) > 0:
            axs[i][1].plot(testdata.transpose(), linewidth=0.5, c=colors[j], alpha=0.2, zorder=np.random.randint(100))


legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=classes[i]) for i in range(len(classes))]
axs[len(bands)][0].legend(handles=legend_elements, ncol=10)

sns.despine(offset=10, left=True)

plt.show()
