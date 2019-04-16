import numpy as np
from utils.classmetric import confusion_matrix_to_accuraccies
import pandas as pd
import os

def confusionmatrix2table(path, classnames=None):
    confusion_matrix = np.load(path)
    overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    support = confusion_matrix.sum(0)

    if classnames is None:
        classnames = np.array(["class_{}".format(i) for i in range(confusion_matrix.shape[0])])

    df = pd.DataFrame([classnames, precision*100, recall*100, f1*100, support.astype(int)]).T
    df.columns = ["names","precision","recall","f1","support"]
    df = df.set_index("names")

    print(df.to_latex(float_format=lambda x: '%10.2f' % x))

def texconfmat(path, classnames=None):
    confmat = np.load(path)

    if classnames is None:
        classnames = np.array(["class_{}".format(i) for i in range(confmat.shape[0])])

    precision = confmat / (confmat.sum(axis=0) + 1e-10)
    recall = confmat / (confmat.sum(axis=1) + 1e-10)

    outcsv = ""
    rows, columns = confmat.shape
    for c in range(columns):
        for r in range(rows):
            row = "{r} {c} {absolute} {precision} {recall}".format(r=r + 1, c=c + 1, absolute=int(confmat[r, c]),
                                                                   precision=precision[r, c], recall=recall[r, c])
            outcsv += row + "\n"

    outfile = os.path.join(os.path.dirname(path),"confmat_flat.csv")
    with open(outfile, "w") as f:
        f.write(outcsv)
    print("writing "+outfile)

if __name__=="__main__":
    confusionmatrix2table("/tmp/test/BavarianCrops/npy/confusion_matrix_1.npy")
    texconfmat("/tmp/test/BavarianCrops/npy/confusion_matrix_1.npy")