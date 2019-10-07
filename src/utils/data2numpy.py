import argparse

import sys
sys.path.append("..")
sys.path.append("../utils")
sys.path.append("../models")

from train import prepare_dataset
from experiments import experiments
import tqdm
import numpy as np

def get_data(experiment, N_per_class=None, N_largest=None, do_add_spectral_indices=True):
    assert experiment in ["isprs_rf_tum","isprs_rf_gaf","isprs_svm_tum","isprs_svm_gaf"]
    assert N_per_class is None or isinstance(N_per_class, int)
    assert N_largest is None or isinstance(N_largest, int)
    assert isinstance(do_add_spectral_indices, bool)

    args = argparse.Namespace(experiment=experiment, seed=0, batchsize=256, workers=0, mode=None, hparamset=0)
    args = experiments(args)

    traindataloader, testdataloader = prepare_dataset(args)

    classnames = traindataloader.dataset.datasets[0].classname

    X, y, ids = dataloader_to_numpy(traindataloader)
    Xtest, ytest, idstest = dataloader_to_numpy(testdataloader)

    if N_largest is not None:
        class_idxs = get_class_idxs(np.hstack([y,ytest]),N_largest)
        X, y, ids = filter_largest(X, y, ids,class_idxs)
        Xtest, ytest, idstest = filter_largest(Xtest, ytest, idstest, class_idxs)
        classnames = classnames[class_idxs]
    else:
        class_idxs = np.arange(len(classnames))

    if N_per_class is not None:
        # make uniform class distributions
        X, y, ids = make_uniform(X, y, ids, N_per_class)
        Xtest, ytest, idstest = make_uniform(Xtest, ytest, idstest, N_per_class)

    # add spectral indices features
    if do_add_spectral_indices:
        X = add_spectral_indices(X)
        Xtest = add_spectral_indices(Xtest)

    return X,y,ids, Xtest, ytest, idstest, classnames, class_idxs

def dataloader_to_numpy(dataloader):
    X = list()
    y = list()
    ids = list()
    for iteration, data in tqdm.tqdm(enumerate(dataloader)):
        inputs, targets, ids_ = [d.cpu().detach().numpy() for d in data]
        X.append(inputs)
        y.append(targets[:, 0])
        ids.append(ids_)

    X = np.vstack(X)
    y = np.hstack(y)
    ids = np.hstack(ids)
    return X, y, ids


def get_uniform_idxs(targets, N_uniform=50):
    classes = np.unique(targets)
    class_idxs = list()
    idxs = np.array([])
    for c in classes:
        idxs_ = np.argwhere(targets == c)[:N_uniform, 0]
        idxs = np.hstack([idxs, idxs_])
    return idxs.astype(int)

def get_class_idxs(y,N_largest):
    count, bins = np.histogram(y, bins=np.unique(y))
    classidxs_by_size = count.argsort()[::-1]
    return classidxs_by_size[:N_largest]

def filter_largest(X, y, ids, class_idxs):
    mask = np.isin(y, class_idxs)
    y = y[mask]
    X = X[mask]
    ids = ids[mask]
    return X, y, ids


def make_uniform(X, y, ids, N_uniform):
    idxs = get_uniform_idxs(y, N_uniform)
    y = y[idxs]
    X = X[idxs]
    ids = ids[idxs]
    return X, y, ids

def X2bands_tum(X):
    # ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9']
    #  0      1       2     3     4     5      6    7     8     9     10   11     12
    nir = X[:, :, 11]
    red = X[:, :, 6]
    green = X[:, :, 5]
    blue = X[:, :, 4]
    b7 = X[:, :, 9]
    b4 = X[:, :, 6]
    b5 = X[:, :, 7]
    b6 = X[:, :, 8]
    b8 = X[:, :, 10]
    b11 = X[:, :, 2]

    return nir, red, green, blue, b4, b5, b6, b7, b8, b11


def X2bands_gaf(X):
    # ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "B8A"]
    #  0      1        2     3       4     5       6     7      8      9
    nir = X[:, :, 9]
    red = X[:, :, 4]
    green = X[:, :, 3]
    blue = X[:, :, 2]
    b7 = X[:, :, 5]
    b4 = X[:, :, 2]
    b5 = X[:, :, 3]
    b6 = X[:, :, 4]
    b8 = X[:, :, 6]
    b11 = X[:, :, 7]

    return nir, red, green, blue, b4, b5, b6, b7, b8, b11


def add_spectral_indices(X):
    # assuming preprocessed gaf dataset
    if X.shape[2] == 10:
        nir, red, green, blue, b4, b5, b6, b7, b8, b11 = X2bands_gaf(X)
    elif X.shape[2] == 13:
        nir, red, green, blue, b4, b5, b6, b7, b8, b11 = X2bands_tum(X)

    ndvi = (nir - red) / (nir + red)

    G = 2.5
    C1 = 6
    C2 = 7.5
    L = 1
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + 1)

    ireci = (b7 - b4) * b6 / b5

    bi = np.sqrt(((red * red) / (green * green)) / 2)
    ndwi = (b8 - b11) / (b8 + b11)

    return np.dstack([X, ndvi, evi, ireci, bi, ndwi])