import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def plot_hyperparameters():
    root = "data/hyperparameters"

    hparams = pd.read_csv("data/hyperparameters/hyperparams_conv1d_v2.csv").set_index("dataset")
    mori = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", delimiter=' ').set_index("Dataset")

    comparison = hparams.join(mori).dropna()
    #comparison["mean_accuracy"] = comparison["mean_accuracy"]*100
    #comparison["eval_accuracy"] = pd.read_csv(os.path.join(root,"eval","eval_e30_cross_entropy.csv"),index_col=0)["accuracy"]

    ours = comparison["mean_accuracy"]*100
    baseline = comparison["a=0.9"]*100
    text = comparison.index
    err = comparison["std_accuracy"]*100

    fig, ax = plt.subplots(figsize=(16,8))
    sns.despine(fig, offset=5)

    ax.errorbar(ours, baseline, xerr=err, fmt='o', alpha=0.5)
    ax.set_xlabel("Accuracy Ours (Phase 1 cross entropy loss only)")
    ax.set_ylabel(r"Accuracy Mori et al. (2017) SR2-CF3 $\alpha=0.9$")
    ax.set_xlim(0,110)
    ax.set_ylim(0,110)
    ax.set_title("Mori et al. (2017) vs Conv1D network with different poolings in valid accuracy")

    # diagonal line
    ax.plot([0,100],[0,100])
    ax.grid()

    for txt, row in comparison.iterrows():
      X = row["mean_accuracy"]*100
      Y = row["a=0.9"]*100
      ax.annotate(txt, (X+0.5, Y+0.5), fontsize=12)

if __name__=="__main__":
    plot_hyperparameters()
    plt.show()