import sys
sys.path.append("../src")
sys.path.append("../src/models")
from tslearn.svm import TimeSeriesSVC
from sklearn.model_selection import RandomizedSearchCV
import scipy
from utils.data2numpy import get_data

def cross_validate(dataset,outfile):

    X,y,ids, _, _, _, classnames, class_idxs = get_data(dataset, N_per_class=500, N_largest=None, do_add_spectral_indices=True)

    n_iter_search = 300

    svm = TimeSeriesSVC(n_jobs=-1)

    random_grid = {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
                  'kernel': ['rbf', 'gak']}

    random_search = RandomizedSearchCV(svm, param_distributions=random_grid,
                                       n_iter=n_iter_search, n_jobs=-1, cv=3, verbose=3)

    random_search.fit(X, y)
    print(random_search.best_params_, )

    rf = random_search.best_estimator_
    print(random_search.best_score_)

    print(str(random_search.best_params_) + " score: " + str(random_search.best_score_), file=open(outfile, "w"))

if __name__=="__main__":
    cross_validate("tum",outfile="/data/isprs/sklearn/svm_tum.txt")
    cross_validate("gaf",outfile="/data/isprs/sklearn/svm_gaf.txt")