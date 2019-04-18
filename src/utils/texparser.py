import numpy as np
from utils.classmetric import confusion_matrix_to_accuraccies
import pandas as pd
import os

def confusionmatrix2table(path, classnames=None, outfile=None):
    confusion_matrix = np.load(path)
    overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    support = confusion_matrix.sum(1) # 0 -> prediction 1 -> ground truth

    if classnames is None:
        classnames = np.array(["class_{}".format(i) for i in range(confusion_matrix.shape[0])])

    df = pd.DataFrame([classnames, list(precision*100), list(recall*100), list(f1*100), list(support.astype(int))]).T

    cols = ["Klasse","Präzision","Genauigkeit","$f_1$-score","\#Felder"]
    df.columns = cols

    df["\#"] = [str(s) for s in np.arange(len(df))+1]
    #df = df.set_index(["\#","Klasse"])

    # add empty row
    df = df.append(pd.Series([np.nan,np.nan,np.nan,np.nan,np.nan], index=cols), ignore_index=True)

    mean_prec = df["Präzision"].mean()
    mean_recall = df["Genauigkeit"].mean()
    mean_fscore = df["$f_1$-score"].mean()
    sum = int(df["\#Felder"].sum())
    df = df.append(pd.Series(["",mean_prec, mean_recall, mean_fscore, sum], index=["Klasse","Präzision","Genauigkeit","$f_1$-score","\#Felder"]), ignore_index=True)

    df = df.set_index(["\#", "Klasse"])

    tex = df.to_latex(float_format=lambda x: '%10.0f' % x, escape=False, na_rep='', column_format="rlcccc")

    print("writing latex tabular to "+outfile)
    print(tex,file=open(outfile, "w"))

def texconfmat(path, classnames=None):
    confmat = np.load(path)

    if classnames is None:
        classnames = np.array(["class_{}".format(i) for i in range(confmat.shape[0])])

    precision = confmat / (confmat.sum(axis=0)[np.newaxis,:] + 1e-10)
    recall = confmat / (confmat.sum(axis=1)[:,np.newaxis] + 1e-10)

    outcsv = ""
    rows, columns = confmat.shape
    for c in range(columns):
        for r in range(rows):
            row = "{r} {c} {absolute} {precision} {recall}".format(r=r + 1, c=c + 1, absolute=np.log10(int(confmat[r, c])),
                                                                   precision=precision[r, c], recall=recall[r, c])
            outcsv += row + "\n"

    outfile = os.path.join(os.path.dirname(path),"confmat_flat.csv")
    with open(outfile, "w") as f:
        f.write(outcsv)
    print("writing "+outfile)

if __name__=="__main__":
    confusionmatrix2table("/tmp/test/BavarianCrops/npy/confusion_matrix_1.npy")
    texconfmat("/tmp/test/BavarianCrops/npy/confusion_matrix_1.npy")