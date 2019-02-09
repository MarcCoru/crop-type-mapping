import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

PLOTS_PATH = "png"
DATA_PATH = "csv"


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

            fname = os.path.join(PLOTS_PATH, "accuracy_{}_{}.png".format(loss ,alpha))
            print("writing  " +fname)
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
            print("writing  " +fname)
            fig.savefig(fname)
            plt.clf()


def plot_earlinessaccuracy(entropy_factor=0.001):

    compare_earliness = pd.read_csv("data/morietal2017/mori-earliness-sr2-cf2.csv", sep=' ').set_index("Dataset")
    compare_accuracy = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")

    # print(compare.columns)

    for alpha in [0.9, 0.8, 0.7, 0.6]:
        for loss in ["twophase_cross_entropy", "twophase_linear_loss"]:

            compare = calc_loss(compare_accuracy["a={}".format(alpha)], 1- compare_earliness["a={}".format(alpha)] * 0.01, alpha)

            csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha,
                                                                         entropy_factor=entropy_factor)

            if not os.path.exists(csvfile):
                print("{} not found. skipping...".format(csvfile))
                continue

            ours = pd.read_csv(csvfile, index_col=0)
            ours["weighted_score"] = calc_loss(ours["phase2_accuracy"], 1 - ours["phase2_earliness"] * 0.01, alpha)

            fig, ax = plot(ours, "weighted_score", pd.DataFrame(compare), "a={}".format(alpha),
                           xlabel="Accuracy Ours (Phase 2)",
                           ylabel=r"Mori et al. (2017) SR2-CF2$",
                           title=r"accuracy and earliness $\alpha={}$".format(alpha))

            fname = os.path.join(PLOTS_PATH, "accuracyearliness_{}_{}.png".format(loss, alpha))
            print("writing " + fname)
            fig.savefig(fname)
            plt.clf()


def phase1_vs_phase2_accuracy():
    all_arrows = True

    col1 = "phase1_accuracy"
    col2 = "phase2_accuracy"

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
                       ylabel=r"accuracy end phase 2 (30 epochs phase1 + 30 epochs phase 2)",
                       title=r"$\alpha={}$".format(alpha),
                       fig=fig,
                       ax=ax,
                       textalpha=0.7)

        fname = os.path.join(PLOTS_PATH, "phase1vs2_accuracy_{}.png".format(alpha))
        print("writing " + fname)

        fig.savefig(fname)
        plt.clf()


def accuracy_vs_earliness():
    all_arrows = True

    col1 = "phase2_accuracy"
    col2 = "phase2_earliness"

    arrowdecay = 0.2

    runs = list()
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        fig, ax = plt.subplots(figsize=(16, 8))

        csvfile = "data/twophase_linear_loss/a{alpha}e0.001.csv".format(alpha=alpha)

        if not os.path.exists(csvfile):
            print("{} not found. skipping...".format(csvfile))
            continue

        run = pd.read_csv(csvfile, index_col=0) * 100

        # insert at beginning
        runs.insert(0, run)

        opacity = 0.0
        for i in range(1, len(runs)):
            opacity += arrowdecay
            if opacity >= 1:
                opacity = 1  #
            if opacity <= 0:
                opacity = 0

            x = runs[i][col1]
            y = runs[i][col2]
            dx = runs[i - 1][col1] - runs[i][col1]
            dy = runs[i - 1][col2] - runs[i][col2]
            for j in range(len(dy)):
                ax.arrow(x[j], y[j], dx[j], dy[j], color="k", alpha=1 - opacity, head_width=0.5)

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
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        ours = pd.read_csv("data/twophase_linear_loss/a{alpha}e0.001.csv".format(alpha=alpha), index_col=0)
        phase1_accuracy = ours["phase1_accuracy"]
        phase1_accuracy.name = "alpha={}".format(alpha)
        merged.append(phase1_accuracy)

    concat = pd.concat(merged, axis=1)
    mean = concat.mean(axis=1)
    std = concat.std(axis=1)

    mean.name = "mean_accuracy"
    std.name = "std_accuracy"

    ours = pd.concat([mean, std], axis=1)

    fig, ax = plot(ours * 100,
                   "mean_accuracy",
                   compare * 100,
                   "a=0.9",
                   errcol="std_accuracy",
                   xlabel="mean and std of five Phase 1 runs",
                   ylabel=r"Mori et al. (2017) SR2-CF2$",
                   title="Our accuracy phase 1")

    fname = os.path.join(PLOTS_PATH, "phase1accuracy.png")
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

if __name__ == "__main__":

    plot_accuracy(entropy_factor=0.01)
    plot_earliness(entropy_factor=0.01)
    plot_earlinessaccuracy(entropy_factor=0.01)
    phase1_vs_phase2_accuracy()
    accuracy_vs_earliness()
    variance_phase1()