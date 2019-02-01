import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

import numpy as np

PLOTS_PATH = "png"
DATA_PATH = "csv"


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


def load_mori(alpha):
    A = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")
    E = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset") * 0.01
    return A["a={}".format(alpha)], E["a={}".format(alpha)]


def load_relclass(alpha):
    file = "data/morietal2017/relclass-{}-gaussian-quadratic-set.csv"

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

    accuracy = pd.read_csv(file.format("accuracy"), sep=' ').set_index("Dataset")[column]  # accuracy is scaled 0-1
    earliness = pd.read_csv(file.format("earliness"), sep=' ').set_index("Dataset")[
                    column] * 0.01  # earliness is scaled 1-100
    return accuracy, earliness


def plot_accuracyearliness_sota_experiment(ptsepsilon=10, entropy_factor=0.01, compare="mori"):
    csvfile = "data/sota_comparison/runs.csv"
    df = pd.read_csv(csvfile)

    meta = pd.read_csv("data/UCR/DataSummary.csv").set_index("Name")

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
            accuracy, earliness = load_relclass(alpha)
        elif compare=="mori":
            accuracy, earliness = load_mori(alpha)

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

def plot_accuracy_sota_experiment(ptsepsilon=10, entropy_factor=0.01, compare="mori"):
    csvfile = "data/sota_comparison/runs.csv"
    data = pd.read_csv(csvfile).set_index("dataset")
    data = data.loc[data["ptsepsilon"] == ptsepsilon]
    data = data.loc[data["entropy_factor"] == entropy_factor]

    meta = pd.read_csv("data/UCR/DataSummary.csv").set_index("Name")


    for alpha in [0.6, 0.7, 0.8, 0.9]:
        df = data.loc[data["earliness_factor"] == alpha]

        summary = pd.DataFrame()
        for objective in ["accuracy", "earliness"]:

            ours = df.loc[df["earliness_factor"]==alpha]
            if len(ours) == 0:
                print("No runs found for a{} b{} e{}... skipping".format(alpha, entropy_factor, ptsepsilon))
                continue

            if compare == "relclass":
                accuracy, earliness = load_relclass(alpha)
            elif compare == "mori":
                accuracy, earliness = load_mori(alpha)

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

def plot_accuracy(entropy_factor=0.001):
    

    compare = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")

    # print(compare.columns)

    for alpha in [0.9, 0.8, 0.7, 0.6]:
        for loss in ["twophase_cross_entropy", "twophase_linear_loss"]:
            csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha,
                                                                                       entropy_factor=entropy_factor)

            if not os.path.exists(csvfile):
                print("{} not found. skipping...".format(csvfile))
                continue

            ours = pd.read_csv(csvfile, index_col=0)
            fig, ax = plot(ours, "phase2_accuracy", compare, "a={}".format(alpha), xlabel="Accuracy Ours (Phase 2)",
                       ylabel=r"Mori et al. (2017) SR2-CF2$", title=r"accuracy $\alpha={}$".format(alpha))

            fname = os.path.join(PLOTS_PATH, "accuracy_{}_{}.png".format(loss,alpha))
            print("writing "+fname)
            fig.savefig(fname)
            plt.clf()

def plot_earliness(entropy_factor=0.001):
    

    compare = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset")

    # print(compare.columns)

    for alpha in [0.9, 0.8, 0.7, 0.6]:
        for loss in ["twophase_cross_entropy", "twophase_linear_loss"]:
            csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha,
                                                                         entropy_factor=entropy_factor)

            if not os.path.exists(csvfile):
                print("{} not found. skipping...".format(csvfile))
                continue

            ours = pd.read_csv(csvfile, index_col=0)
            fig, ax = plot(ours, "phase2_earliness", compare, "a={}".format(alpha), xlabel="Accuracy Ours (Phase 2)",
                       ylabel=r"Mori et al. (2017) SR2-CF2$", title=r"earliness $\alpha={}$".format(alpha))

            fname = os.path.join(PLOTS_PATH, "earliness_{}_{}.png".format(loss, alpha))
            print("writing "+fname)
            fig.savefig(fname)
            plt.clf()


def calc_loss(accuracy,earliness,alpha):
    return alpha * accuracy + (1 - alpha) * (earliness)

def plot_earlinessaccuracy(entropy_factor=0.001):
    

    compare_earliness = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset")
    compare_accuracy = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")

    # print(compare.columns)

    for alpha in [0.9, 0.8, 0.7, 0.6]:
        for loss in ["twophase_cross_entropy", "twophase_linear_loss"]:

            compare = calc_loss(compare_accuracy["a={}".format(alpha)], 1-compare_earliness["a={}".format(alpha)]*0.01, alpha)

            csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha,
                                                                         entropy_factor=entropy_factor)

            if not os.path.exists(csvfile):
                print("{} not found. skipping...".format(csvfile))
                continue

            ours = pd.read_csv(csvfile, index_col=0)
            ours["weighted_score"] = calc_loss(ours["phase2_accuracy"],1-ours["phase2_earliness"]*0.01,alpha)


            fig, ax = plot(ours, "weighted_score", pd.DataFrame(compare), "a={}".format(alpha), xlabel="Accuracy Ours (Phase 2)",
                       ylabel=r"Mori et al. (2017) SR2-CF2$", title=r"accuracy and earliness $\alpha={}$".format(alpha))

            fname = os.path.join(PLOTS_PATH, "accuracyearliness_{}_{}.png".format(loss, alpha))
            print("writing "+fname)
            fig.savefig(fname)
            plt.clf()


def phase1_vs_phase2_accuracy():
    

    all_arrows=True

    col1="phase1_accuracy"
    col2="phase2_accuracy"

    runs = list()
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fig, ax = plt.subplots(figsize=(16, 8))

        csvfile = "data/twophase_linear_loss/a{alpha}e0.001.csv".format(alpha=alpha)

        if not os.path.exists(csvfile):
            print("{} not found. skipping...".format(csvfile))
            continue

        run = pd.read_csv(csvfile, index_col=0) * 100

        runs.append(run)

        if all_arrows:
            for i in range(1, len(runs)):
                x = runs[i][col1]
                y = runs[i][col2]
                dx = runs[i - 1][col1] - runs[i][col1]
                dy = runs[i - 1][col2] - runs[i][col2]
                for j in range(len(dy)):
                    ax.arrow(x[j], y[j], dx[j], dy[j], color="k", alpha=0.4)
        else:
            x = runs[-1][col1]
            y = runs[-1][col2]
            dx = runs[0][col1] - runs[-1][col1]
            dy = runs[0][col2] - runs[-1][col2]
            for j in range(len(dy)):
                ax.arrow(x[j], y[j], dx[j], dy[j], color="k", alpha=0.4)

        fig, ax = plot(run, col1, run, col2,
                    xlabel="accuracy end phase 1  (30 epochs cross entropy)",
                    ylabel=r"accuracy end phase 2 (30 epochs phase1 + 30 epochs phase 2)", title=r"$\alpha={}$".format(alpha),
                    fig=fig,
                    ax=ax,
                    textalpha=0.7)

        fname = os.path.join(PLOTS_PATH, "phase1vs2_accuracy_{}.png".format(alpha))
        print("writing " + fname)

        fig.savefig(fname)
        plt.clf()

def accuracy_vs_earliness():
    

    all_arrows=True

    col1="phase2_accuracy"
    col2="phase2_earliness"

    arrowdecay=0.2

    runs = list()
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        fig, ax = plt.subplots(figsize=(16, 8))

        csvfile = "data/twophase_linear_loss/a{alpha}e0.001.csv".format(alpha=alpha)

        if not os.path.exists(csvfile):
            print("{} not found. skipping...".format(csvfile))
            continue

        run = pd.read_csv(csvfile, index_col=0) * 100

        # insert at beginning
        runs.insert(0,run)

        opacity = 0.0
        for i in range(1, len(runs)):
            opacity += arrowdecay
            if opacity>=1:
                opacity=1#
            if opacity<=0:
                opacity=0

            x = runs[i][col1]
            y = runs[i][col2]
            dx = runs[i - 1][col1] - runs[i][col1]
            dy = runs[i - 1][col2] - runs[i][col2]
            for j in range(len(dy)):
                ax.arrow(x[j], y[j], dx[j], dy[j], color="k", alpha=1-opacity,head_width=0.5)

        fig, ax = plot(run, col1, run, col2,
                    xlabel="Accuracy",
                    ylabel=r"Earliness", title=r"$\alpha={}$".format(alpha),
                    fig=fig,
                    ax=ax,
                    textalpha=0.7,
                    diagonal=False)

        fname = os.path.join(PLOTS_PATH, "accuracy_vs_earliness_{}.png".format(alpha))
        print("writing " + fname)

        fig.savefig(fname)
        plt.clf()


def variance_phase1():
    


    compare = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")

    merged = list()
    for alpha in [1.0,0.9, 0.8, 0.7, 0.6, 0.5]:
        ours = pd.read_csv("data/twophase_linear_loss/a{alpha}e0.001.csv".format(alpha=alpha), index_col=0)
        phase1_accuracy = ours["phase1_accuracy"]
        phase1_accuracy.name = "alpha={}".format(alpha)
        merged.append(phase1_accuracy)

    concat = pd.concat(merged, axis=1)
    mean = concat.mean(axis=1)
    std = concat.std(axis=1)

    mean.name = "mean_accuracy"
    std.name = "std_accuracy"

    ours = pd.concat([mean,std],axis=1)

    fig, ax = plot(ours * 100,
                   "mean_accuracy",
                   compare * 100,
                   "a=0.9",
                   errcol="std_accuracy",
                   xlabel = "mean and std of five Phase 1 runs",
                   ylabel = r"Mori et al. (2017) SR2-CF2$",
                   title = "Our accuracy phase 1")

    fname = os.path.join(PLOTS_PATH, "phase1accuracy.png")
    print("writing " + fname)
    fig.savefig(fname)
    plt.clf()

def qualitative_figure():

    csvfile = "data/sota_comparison/runs.csv"
    df = pd.read_csv(csvfile)
    df = df.loc[df.entropy_factor == 0]
    df = df.loc[df.ptsepsilon == 0]


    mori_accuracy = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")
    mori_earliness = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset")

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

def load_approaches(alpha=0.6,relclass_col="t=0.001",edsc_col="t=2.5",ects_col="sup=0.05"):
    def load(file, column, name):
        accuracy = pd.read_csv(file.format("accuracy"), sep=' ').set_index("Dataset")[column]  # accuracy is scaled 0-1
        accuracy.name = name + "_accuracy"
        earliness = pd.read_csv(file.format("earliness"), sep=' ').set_index("Dataset")[
                        column] * 0.01  # earliness is scaled 1-100
        earliness.name = name + "_earliness"
        return pd.concat([accuracy, earliness], axis=1)

    mori = load("data/morietal2017/mori-{}-sr2-cf2.csv","a={}".format(alpha), "mori")
    relclass = load("data/morietal2017/relclass-{}-gaussian-quadratic-set.csv",relclass_col, "relclass")
    edsc = load("data/morietal2017/edsc-{}.csv",edsc_col, "edsc")
    ects = load("data/morietal2017/ects-{}-strict-method.csv",ects_col, "ects")

    return pd.concat([mori,relclass,edsc,ects], axis=1, join="inner")

def qualitative_figure_single_dataset():
    dataset = "TwoPatterns"

    csvfile = "data/sota_comparison/runs.csv"
    df = pd.read_csv(csvfile)

    fig, ax = plt.subplots(figsize=(16, 8))

    accuracy = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")
    earliness = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset")
    accuracy = accuracy.loc[dataset]
    accuracy.name = "accuracy"
    earliness = earliness.loc[dataset]
    earliness.name = "earliness"

    mori = pd.concat([accuracy, earliness * 0.01], axis=1)

    earliness_factors = list()
    for index, row in mori.iterrows():
        earliness_factors.append(float(index.split("=")[-1]))
    mori["earliness_factor"] = earliness_factors
    mori = mori.set_index("earliness_factor")

    ours = df.loc[df["dataset"] == dataset].sort_values(by="earliness_factor").set_index("earliness_factor")

    for dataframe in [mori, ours]:
        ax.plot(dataframe["accuracy"], dataframe["earliness"], linestyle='--', marker='o')

    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("accuracy")
    ax.set_ylabel("earliness")


    fname = os.path.join(PLOTS_PATH, "earlinessaccuracy.png")
    print("writing " + fname)
    fig.savefig(fname)
    plt.clf()

if __name__=="__main__":

    #plot_accuracy(entropy_factor=0.01)
    #plot_earliness(entropy_factor=0.01)
    #plot_earlinessaccuracy(entropy_factor=0.01)
    #phase1_vs_phase2_accuracy()
    #accuracy_vs_earliness()
    #variance_phase1()

    if False:
        for compare in ["mori","relclass"]:

            for ptsepsilon in [0,5,10,50]:
                for entropy_factor in [0,0.01, 0.1]:
                    plot_accuracyearliness_sota_experiment(ptsepsilon=ptsepsilon, entropy_factor=entropy_factor, compare=compare)
                    plot_accuracy_sota_experiment(ptsepsilon=ptsepsilon, entropy_factor=entropy_factor, compare=compare)

        plot_accuracyearliness_sota_experiment(ptsepsilon=5, entropy_factor=0.01)
        plot_accuracy_sota_experiment(ptsepsilon=5, entropy_factor=0.01)

    qualitative_figure()

