import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
sns.set_style("white")

import numpy as np

PLOTS_PATH = "png"
DATA_PATH = "csv"

def calc_loss(accuracy ,earliness ,alpha):
    return alpha * accuracy + (1 - alpha) * (earliness)

def plot(df_a, col_a, df_b, col_b, xlabel="", ylabel="", title="",
         fig=None, ax=None, textalpha=1, errcol=None, diagonal=True, hue=None, cbar_label=""):

    if df_a.equals(df_b):
        concated = df_a
    else:
        concated = pd.concat([df_a, df_b], axis=1, join='inner')

    if hue is not None:
        hue.name = "color"
        hue.index.names = ['dataset']
        concated["color"] = hue

        # features in consistent sequence to datasets
        hue = concated["color"]

    X = concated[col_a]
    Y = concated[col_b]
    if errcol is not None:
        err = concated[errcol]
    else:
        err = None
    text = df_a.index

    if fig is None:
        fig, ax = plt.subplots(figsize=(16, 8))

    sns.despine(fig, offset=5)

    if err is None:
        sc = ax.scatter(X, Y, c=hue)

        if hue is not None:
            cbar = plt.colorbar(sc)
            cbar.set_label(cbar_label, rotation=270)
    else:
        ax.errorbar(X, Y, xerr=err, fmt='o', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)

    if diagonal:
        # diagonal line
        ax.plot([0, 1], [0, 1])
    ax.grid()

    xy = pd.concat([X, Y], axis=1)
    xy.columns = ["X", "Y"]

    for row in xy.iterrows():
        txt, (xval, yval) = row
        ax.annotate(txt, (xval + 0.01, yval + 0.01), fontsize=12, alpha=textalpha)

    return fig, ax

def load_mori(alpha, mori_accuracy, mori_earliness):
    A = pd.read_csv(mori_accuracy, sep=' ').set_index("Dataset")
    E = pd.read_csv(mori_earliness, sep=' ').set_index("Dataset") * 0.01
    return A["a={}".format(alpha)], E["a={}".format(alpha)]

def load_relclass(alpha, relclass_accuracy, relclass_earliness):

    if alpha == 0.6:
        column = "t=0.001"
    elif alpha == 0.7:
        column = "t=0.1"
    elif alpha == 0.8:
        column = "t=0.5"
    elif alpha == 0.9:
        column = "t=0.9"
    else:
        raise ValueError()

    accuracy = pd.read_csv(relclass_accuracy, sep=' ').set_index("Dataset")[column]  # accuracy is scaled 0-1
    earliness = pd.read_csv(relclass_earliness, sep=' ').set_index("Dataset")[
                    column] * 0.01  # earliness is scaled 1-100
    return accuracy, earliness

def plot_accuracyearliness_sota_experiment(ptsepsilon=10,
                                           entropy_factor=0.01,
                                           compare="mori",
                                           csvfile = "data/sota_comparison/runs.csv",
                                           metafile="data/UCR_Datasets/DataSummary.csv",
                                           mori_accuracy="data/morietal2017/mori-accuracy-sr2-cf2.csv",
                                           mori_earliness="data/morietal2017/mori-earliness-sr2-cf2.csv",
                                           relclass_accuracy="data/morietal2017/relclass-accuracy-gaussian-quadratic-set.csv",
                                           relclass_earliness="data/morietal2017/relclass-earliness-gaussian-quadratic-set.csv"):
    df = pd.read_csv(csvfile)

    meta = pd.read_csv(metafile).set_index("Name")

    for alpha in [0.6, 0.7, 0.8, 0.9]:
        data = df.loc[df["earliness_factor"] == alpha].set_index("dataset")
        data = data.loc[data["ptsepsilon"] == ptsepsilon]
        data = data.loc[data["entropy_factor"] == entropy_factor]

        if len(data) == 0:
            print("No runs found for a{} b{} e{}... skipping".format(alpha, entropy_factor, ptsepsilon))
            continue

        cost = calc_loss(data["accuracy"],1-data["earliness"],alpha)

        # average multiple runs
        data["ours"] = cost.groupby(cost.index).mean()
        #load_approaches



        #df = load_approaches(alpha=0.6,relclass_col="t=0.001",edsc_col="t=2.5",ects_col="sup=0.05")
        if compare=="relclass":
            accuracy, earliness = load_relclass(alpha, relclass_accuracy, relclass_earliness)
        elif compare=="mori":
            accuracy, earliness = load_mori(alpha, mori_accuracy, mori_earliness)

        data["mori"] = calc_loss(accuracy,1-earliness,alpha)

        fig, ax = plot(data, "ours", data, "mori", xlabel="accuracy and earliness Ours (Phase 2)",
                       ylabel=compare + r"SR2-CF2$", hue=np.log10(meta["Train"]),
                       cbar_label="log # training samples",
                       title=r"accuracy and earliness $\alpha={}$".format(alpha))

        fname = os.path.join(PLOTS_PATH,compare, "sota_{}_a{}_b{}_e{}.png".format("accuracyearliness", alpha, entropy_factor, ptsepsilon))
        os.makedirs(os.path.dirname(fname),exist_ok=True)
        print("writing " + fname)
        fig.savefig(fname)
        plt.clf()

        fname = os.path.join(DATA_PATH, compare,"accuracyearliness_alpha{}.csv".format(alpha))
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        print("writing " + fname)
        data[["ours", "mori"]].to_csv(fname)

def plot_accuracy_sota_experiment(ptsepsilon=10,
                                  entropy_factor=0.01,
                                  compare="mori",
                                  csvfile = "data/sota_comparison/runs.csv",
                                  metafile="data/UCR_Datasets/DataSummary.csv",
                                  mori_accuracy="data/morietal2017/mori-accuracy-sr2-cf2.csv",
                                  mori_earliness="data/morietal2017/mori-earliness-sr2-cf2.csv",
                                  relclass_accuracy="data/morietal2017/relclass-accuracy-gaussian-quadratic-set.csv",
                                  relclass_earliness="data/morietal2017/relclass-earliness-gaussian-quadratic-set.csv"):

    data = pd.read_csv(csvfile).set_index("dataset")
    data = data.loc[data["ptsepsilon"] == ptsepsilon]
    data = data.loc[data["entropy_factor"] == entropy_factor]

    meta = pd.read_csv(metafile).set_index("Name")


    for alpha in [0.6, 0.7, 0.8, 0.9]:
        df = data.loc[data["earliness_factor"] == alpha]

        summary = pd.DataFrame()
        for objective in ["accuracy", "earliness"]:

            ours = df.loc[df["earliness_factor"]==alpha]
            if len(ours) == 0:
                print("No runs found for a{} b{} e{}... skipping".format(alpha, entropy_factor, ptsepsilon))
                continue

            if compare == "relclass":
                accuracy, earliness = load_relclass(alpha, relclass_accuracy, relclass_earliness)
            elif compare == "mori":
                accuracy, earliness = load_mori(alpha, mori_accuracy, mori_earliness)

            other = pd.DataFrame()
            if objective == "accuracy":
                ours["accuracy"] = ours["accuracy"]
                other["a={}".format(alpha)] = accuracy
                summary["ours"] = ours["accuracy"]
                summary["mori"] = accuracy

            elif objective == "earliness":
                other["a={}".format(alpha)] = earliness
                ours["earliness"] = ours["earliness"]
                summary["mori"] = other

            fig, ax = plot(ours, objective, other, "a={}".format(alpha), xlabel=objective+" Ours (Phase 2)",
                           ylabel=compare + r" SR2-CF2$", title=objective + r" $\alpha={}$".format(alpha),
                           hue=np.log10(meta["Train"]), cbar_label="log # training samples")

            fname = os.path.join(PLOTS_PATH, compare, "sota_{}_a{}_b{}_e{}.png".format(objective,alpha, entropy_factor, ptsepsilon))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print("writing " + fname)
            fig.savefig(fname)
            plt.clf()

            fname = os.path.join(DATA_PATH,compare,"{}_alpha{}.csv".format(objective,alpha))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print("writing " + fname)
            summary[["ours", "mori"]].to_csv(fname)


def qualitative_figure(csvfile = "data/sota_comparison/runs.csv", compare_accuracy="data/morietal2017/mori-accuracy-sr2-cf2.csv", compare_earliness="data/morietal2017/mori-earliness-sr2-cf2.csv"):

    df = pd.read_csv(csvfile)
    df = df.loc[df.entropy_factor == 0]
    df = df.loc[df.ptsepsilon == 0]


    mori_accuracy = pd.read_csv(compare_accuracy, sep=' ').set_index("Dataset")
    mori_earliness = pd.read_csv(compare_earliness, sep=' ').set_index("Dataset")

    all_datasets = df["dataset"].unique()
    #selected_datasets = ["TwoPatterns", "Yoga", "Adiac", "UWaveGestureLibraryY", "FaceAll"]
    selected_datasets = all_datasets

    mori = pd.DataFrame()
    for dataset in selected_datasets:

        fig, ax = plt.subplots(figsize=(30, 16))

        mori["accuracy"] = mori_accuracy.loc[dataset]
        mori["earliness"] = mori_earliness.loc[dataset] * 0.01
        ax.plot(mori["accuracy"], mori["earliness"], linestyle='--', marker='+')

        #df3 = df3[~df3.index.duplicated(keep='first')]
        sample = df.loc[df["dataset"] == dataset].sort_values(by="earliness_factor")
        sample = sample.groupby("earliness_factor").mean()
        ax.plot(sample["accuracy"], sample["earliness"], linestyle='--', marker='o')
        ax.set_xlabel("accuracy")
        ax.set_ylabel("earliness")
        #

        X = sample["accuracy"].iloc[0]
        Y = sample["earliness"].iloc[0]
        ax.annotate("our " + dataset, xy=(X, Y), xytext=(X, Y))

        X = mori["accuracy"].iloc[0]
        Y = mori["earliness"].iloc[0]
        ax.annotate(dataset, xy=(X, Y), xytext=(X, Y))

        fname = DATA_PATH + "/alphas_our_{}.csv".format(dataset)
        print("writing " + fname)
        sample.to_csv(fname)

        fname = DATA_PATH + "/alphas_mori_{}.csv".format(dataset)
        print("writing " + fname)
        mori.to_csv(fname)

        fname = os.path.join(PLOTS_PATH, "earlinessaccuracy", "{}.png".format(dataset))
        os.makedirs(os.path.dirname(fname),exist_ok=True)
        print("writing " + fname)
        fig.savefig(fname)
        plt.clf()

def plot_scatter(csvfile="data/sota_comparison/runs.csv",
                 metafile="data/UCR_Datasets/DataSummary.csv",
                 mori_accuracy="data/morietal2017/mori-accuracy-sr2-cf2.csv",
                 mori_earliness="data/morietal2017/mori-earliness-sr2-cf2.csv",
                 relclass_accuracy="data/morietal2017/relclass-accuracy-gaussian-quadratic-set.csv",
                 relclass_earliness="data/morietal2017/relclass-earliness-gaussian-quadratic-set.csv"):

    for compare in ["mori","relclass"]:

        for ptsepsilon in [0,5,10,50]:
            for entropy_factor in [0,0.01, 0.1]:
                plot_accuracyearliness_sota_experiment(ptsepsilon=ptsepsilon,
                                                       entropy_factor=entropy_factor,
                                                       compare=compare,
                                                       csvfile=csvfile,
                                                       metafile=metafile,
                                                       mori_accuracy=mori_accuracy,
                                                       mori_earliness=mori_earliness,
                                                       relclass_accuracy=relclass_accuracy,
                                                       relclass_earliness=relclass_earliness)

                plot_accuracy_sota_experiment(ptsepsilon=ptsepsilon,
                                              entropy_factor=entropy_factor,
                                              compare=compare,
                                              csvfile=csvfile,
                                              metafile=metafile,
                                              mori_accuracy=mori_accuracy,
                                              mori_earliness=mori_earliness,
                                              relclass_accuracy=relclass_accuracy,
                                              relclass_earliness=relclass_earliness)

def cleanup():
    thisdir=os.path.dirname(os.path.realpath(__file__))

    if os.path.exists(os.path.join(thisdir, DATA_PATH)):
        print("deleting csv data in "+os.path.join(thisdir, DATA_PATH))
        shutil.rmtree(os.path.join(thisdir, DATA_PATH))
    if os.path.exists(os.path.join(thisdir, PLOTS_PATH)):
        print("deleting png plots in " + os.path.join(thisdir, PLOTS_PATH))
        shutil.rmtree(os.path.join(thisdir, PLOTS_PATH))

if __name__=="__main__":
    cleanup()

    csvfile = "../data/runs_conv1d.csv"
    metafile = "../data/UCR_Datasets/DataSummary.csv"
    mori_accuracy = "../data/UCR_Datasets/mori-accuracy-sr2-cf2.csv"
    mori_earliness = "../data/UCR_Datasets/mori-earliness-sr2-cf2.csv"
    relclass_accuracy = "../data/UCR_Datasets/relclass-accuracy-gaussian-quadratic-set.csv"
    relclass_earliness = "../data/UCR_Datasets/relclass-earliness-gaussian-quadratic-set.csv"

    plot_scatter(csvfile, metafile, mori_accuracy, mori_earliness, relclass_accuracy, relclass_earliness)

    qualitative_figure(csvfile,
                       compare_accuracy=mori_accuracy,
                       compare_earliness=mori_earliness)

