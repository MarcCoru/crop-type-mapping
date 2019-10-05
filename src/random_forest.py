
import sys
sys.path.append("..")
sys.path.append("../models")
sys.path.append("../utils")

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from utils.data2numpy import get_data


def flatten(x):
    return x.reshape(x.shape[0], -1)

def cross_validate(dataset,outfile):

    X,y,ids, Xtest, ytest, idstest, classnames, class_idxs = get_data(dataset, N_per_class=500, N_largest=None, do_add_spectral_indices=True)

    n_iter_search = 300

    rf = RandomForestClassifier()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = sp_randint(2, 10)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = sp_randint(1, 4)
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    random_search = RandomizedSearchCV(rf, param_distributions=random_grid,
                                       n_iter=n_iter_search, n_jobs=-1, cv=3, verbose=3)

    random_search.fit(flatten(X), y)
    print(random_search.best_params_, )

    rf = random_search.best_estimator_
    print(random_search.best_score_)

    print(str(random_search.best_params_) + " score: " + str(random_search.best_score_), file=open(outfile, "w"))

if __name__=="__main__":
    cross_validate("tum",outfile="/data/isprs/sklearn/random_forest_tum.txt")
    cross_validate("gaf",outfile="/data/isprs/sklearn/random_forest_gaf.txt")
