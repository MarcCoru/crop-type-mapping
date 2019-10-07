import sys
sys.path.append("../src")
sys.path.append("../src/models")
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import RandomizedSearchCV
import scipy
from utils.data2numpy import get_data
import pandas as pd

def cross_validate(experiment,outfile):

    X,y,ids, _, _, _, classnames, class_idxs = get_data(experiment, N_per_class=500, N_largest=None, do_add_spectral_indices=True)

    n_iter_search = 1

    svm = TimeSeriesSVC(n_jobs=-1)

    random_grid = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                  'kernel': ['rbf']}

    random_search = RandomizedSearchCV(svm, param_distributions=random_grid,scoring='f1_macro',
                                       n_iter=n_iter_search, n_jobs=-1, cv=3, verbose=3)

    random_search.fit(X, y)
    print(random_search.best_params_, )

    print(random_search.best_score_)
    print(random_search.cv_results_)



    print(str(random_search.best_params_) + " score: " + str(random_search.best_score_), file=open(outfile, "w"))

    df = pd.DataFrame(random_search.cv_results_)
    print(f"writing {outfile}.csv")
    df.to_csv(outfile + ".csv")


if __name__=="__main__":
    cross_validate("isprs_svm_tum",outfile="/data/isprs/sklearn/svm_tum.txt")
    cross_validate("isprs_svm_gaf",outfile="/data/isprs/sklearn/svm_gaf.txt")
