import numpy as np
from utils.classmetric import confusion_matrix_to_accuraccies
import pandas as pd
import os

def confusionmatrix2table(path, ids=None, classnames=None, outfile=None):
    confusion_matrix = np.load(path)
    overall_accuracy, kappa, precision, recall, f1, cl_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    support = confusion_matrix.sum(1) # 0 -> prediction 1 -> ground truth

    if classnames is None:
        classnames = np.array(["{}".format(i) for i in range(confusion_matrix.shape[0])])

    df = pd.DataFrame([ids, classnames, list(precision*100), list(recall*100), list(f1*100), list(support.astype(int))]).T

    cols = ["ID", "Klasse","Präzision","Genauigkeit","$f_1$-score","\#Felder"]
    df.columns = cols

    #df["ID"] = ids
    #df = df.set_index(["ID"])

    # add empty row
    df = df.append(pd.Series([np.nan,np.nan,np.nan,np.nan,np.nan, np.nan], index=cols), ignore_index=True)

    mean_prec = df["Präzision"].mean()
    mean_recall = df["Genauigkeit"].mean()
    mean_fscore = df["$f_1$-score"].mean()
    sum = int(df["\#Felder"].sum())
    df = df.append(pd.Series(["per Klasse",mean_prec, mean_recall, mean_fscore, sum], index=["Klasse","Präzision","Genauigkeit","$f_1$-score","\#Felder"]), ignore_index=True)
    # add empty row
    df = df.append(pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=cols), ignore_index=True)
    df = df.append(pd.Series(["per Parzelle", "kappa", "overall accuracy"], index=["Klasse", "Präzision", "Genauigkeit"]),  ignore_index=True)
    df = df.append(pd.Series([kappa, overall_accuracy*100], index=["Präzision", "Genauigkeit"]),ignore_index=True)

    df = df.set_index(["ID", "Klasse"])

    tex = df.to_latex(float_format=lambda x: '%10.2f' % x, escape=False, na_rep='', column_format="rlcccc")

    print("writing latex tabular to "+outfile)
    print(tex,file=open(outfile, "w"))

def texconfmat(path, classnames=None, outfile=None):
    confmat = np.load(path)

    precision = confmat / (confmat.sum(axis=0)[np.newaxis,:] + 1e-10)
    recall = confmat / (confmat.sum(axis=1)[:,np.newaxis] + 1e-10)

    outcsv = ""
    rows, columns = confmat.shape
    for c in range(columns):
        for r in range(rows):

            if np.log10(int(confmat[r, c])) == -np.inf:
                abs = 0
            else:
                abs = np.log10(int(confmat[r, c]))

            row = "{r} {c} {absolute} {precision} {recall}".format(r=r + 1, c=c + 1, absolute=abs,
                                                                   precision=precision[r, c], recall=recall[r, c])
            outcsv += row + "\n"

    #outfile = os.path.join(os.path.dirname(path),"confmat_flat.csv")
    with open(outfile, "w") as f:
        f.write(outcsv)
    print("writing "+outfile)

def load_run(path):
    run = pd.read_csv(path)
    run = run.loc[run["mode"]=="test"]
    return run

def parse_run(run, classnapping, outdir):
    root=run
    mapping = pd.read_csv(classnapping)


    code = mapping.gafcode.unique()
    name = mapping.groupby("gafcode").first().klassenname


    try:
        run = load_run(os.path.join(root, "log.csv"))
        best_epoch = run.sort_values(by="kappa", ascending=False).iloc[0].epoch
        cm = "confusion_matrix_{}.npy".format(best_epoch)
        print(cm)
        confusionmatrix2table(os.path.join(root,"npy",cm), outfile=os.path.join(outdir,"table.tex"), ids=code, classnames=name)
        texconfmat(os.path.join(root,"npy",cm), outfile=os.path.join(outdir,"confmat_flat.csv"), classnames=code)
    except:
        print("could not write "+os.path.join(outdir,"table.tex"))

if __name__=="__main__":


    import os
    root = "/data/gaf/runs/tumgaf_gaf_rnn"
    classmapping="/data/BavarianCrops/classmapping.csv.gaf.v2"
    parse_run(root, classmapping, outdir="/data/gaf/runs/tumgaf_gaf_rnn")


