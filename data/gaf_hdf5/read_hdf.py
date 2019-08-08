import re
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hdf =  pd.HDFStore('./test_train.h5','r')
testset =  hdf['test_data']
trainset =  hdf['train_data']
hdf.close()

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

def get_data(band, type="raw"):
    return trainset[list(categories[band][type])], testset[list(categories[band][type])]



pass

colors = 2 * ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

fig, axs = plt.subplots(2,1)
for i in range(len(bands)):
    train, test = get_data(bands[i], "raw")
    axs[0].plot(train.iloc[:50].transpose().values, linewidth=0.2, c=colors[i])
    axs[1].plot(test.iloc[:50].transpose().values, linewidth=0.2, c=colors[i])
    axs[1].legend()

plt.show()