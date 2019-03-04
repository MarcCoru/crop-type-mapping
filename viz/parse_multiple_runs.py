import pandas as pd
import os

root = "/data/EV2019"
csv = "/home/marc/projects/EV2019/images/logs/data/early_reward/alphas.csv"

epoch=100

def parse_multiple_runs_single(runs, replacestr = "earlyrewarda"):

    runpaths = [os.path.join(root,run) for run in runs]

    last_results = list()
    for runpath in runpaths:
        data = pd.read_csv(os.path.join(runpath,"BavarianCrops","log_earliness.csv"))
        series = (data.loc[data["mode"] == "test"].set_index("epoch").loc[epoch])
        series["alpha"] = float(os.path.basename(runpath).replace(replacestr,""))
        os.path.basename(runpath)
        last_results.append(series)

    df = pd.DataFrame(last_results).set_index("alpha")

    print("writing "+csv)
    df.to_csv(csv)

    print(r"alpha & accuracy & earliness & f1 & precision & recall & kappa \\")
    for index, row in df.iterrows():

        print(r"{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\".format(index, row["accuracy"], row["earliness"], row["mean_f1"], row["mean_precision"], row["mean_recall"], row["kappa"]))

runs = [
    "earlyrewarda0",
    "earlyrewarda0.2",
    "earlyrewarda0.4",
    "earlyrewarda0.6",
    "earlyrewarda0.8",
    "earlyrewarda1"
]

parse_multiple_runs_single(runs, replacestr="earlyrewarda")

runs = [
    "twophasecrossentropy0",
    "twophasecrossentropy0.2",
    "twophasecrossentropy0.4",
    "twophasecrossentropy0.6",
    "twophasecrossentropy0.8",
    "twophasecrossentropy1"
]

parse_multiple_runs_single(runs, replacestr="twophasecrossentropy")

def parse_multiple_runs(path, alphas=[0,0.2,0.4,0.6,0.8,1],runs=[0,1,2]):

    last_results = list()
    for alpha in alphas:
        for run in runs:
            runpath = os.path.join(path + "-alpha{}-run{}".format(alpha,run))
            data = pd.read_csv(os.path.join(runpath, "BavarianCrops", "log_earliness.csv"))
            testdata = data.loc[data["mode"] == "test"]
            series = testdata.loc[testdata["epoch"]==epoch].iloc[-1]
            series["alpha"] = alpha
            series["run"] = run
            last_results.append(series)


    df = pd.DataFrame(last_results)


    std = df.groupby(by="alpha").std()
    mean = df.groupby(by="alpha").mean()
    #print("writing "+csv)
    #df.to_csv(csv)
    cols = ["accuracy", "earliness", "mean_f1", "mean_precision", "mean_recall", "kappa"]
    for col in cols:
        mean[col+"_std"] = std[col]

    # scale to %
    for col in ["accuracy", "earliness", "mean_f1", "mean_precision", "mean_recall"]:
        mean[col] = mean[col]
        mean[col+"_std"] = mean[col+"_std"]

    print(r"alpha & accuracy & earliness & f1 & precision & recall & kappa \\")
    for index, row in mean.iterrows():

        line = r"{} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} \\".format(
            index,
            row["accuracy"], row["accuracy_std"],
            row["earliness"], row["earliness_std"],
            row["mean_f1"], row["mean_f1_std"],
            row["mean_precision"], row["mean_precision_std"],
            row["mean_recall"], row["mean_recall_std"],
            row["kappa"], row["kappa_std"])
        print(line.replace("0.", "."))

print()
print("earlyreward")
parse_multiple_runs(path="/data/EV2019/earlyreward")

print()
print("twophasecrossentropy")
parse_multiple_runs(path="/data/EV2019/twophasecrossentropy")

def parse_multiple_runs_epsilon(path, alphas=[0,0.2,0.4,0.6,0.8,1],runs=[0], epsilons=[0,1,10], groupcol="alpha"):

    last_results = list()
    for alpha in alphas:
        for epsilon in epsilons:
            for run in runs:
                runpath = os.path.join(path + "-alpha{}-epsilon{}-run{}".format(alpha,epsilon,run))
                data = pd.read_csv(os.path.join(runpath, "BavarianCrops", "log_earliness.csv"))
                testdata = data.loc[data["mode"] == "test"]
                series = testdata.loc[testdata["epoch"]==epoch].iloc[-1]
                series["alpha"] = alpha
                series["epsilon"] = epsilon
                series["run"] = run
                last_results.append(series)


    df = pd.DataFrame(last_results)


    std = df.groupby(by=groupcol).std()
    mean = df.groupby(by=groupcol).mean()
    #print("writing "+csv)
    #df.to_csv(csv)
    cols = ["accuracy", "earliness", "mean_f1", "mean_precision", "mean_recall", "kappa"]
    for col in cols:
        mean[col+"_std"] = std[col]

    # scale to %
    for col in ["accuracy", "earliness", "mean_f1", "mean_precision", "mean_recall"]:
        mean[col] = mean[col]
        mean[col+"_std"] = mean[col+"_std"]

    print(groupcol+r" & accuracy & earliness & f1 & precision & recall & kappa \\")
    for index, row in mean.iterrows():

        line = r"{} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} \\".format(
            index,
            row["accuracy"], row["accuracy_std"],
            row["earliness"], row["earliness_std"],
            row["mean_f1"], row["mean_f1_std"],
            row["mean_precision"], row["mean_precision_std"],
            row["mean_recall"], row["mean_recall_std"],
            row["kappa"], row["kappa_std"])
        print(line.replace("0.", "."))

for alpha in [0,0.2,0.4,0.6,0.8,1]:
    print()
    print("alpha={} earlyreward".format(alpha))
    parse_multiple_runs_epsilon(path="/data/EV2019/earlyreward",groupcol="epsilon", alphas=[alpha])

    #print()
    #print("alpha={} twophasecrossentropy".format(alpha))
    #parse_multiple_runs_epsilon(path="/data/EV2019/twophasecrossentropy",groupcol="epsilon", alphas=[0.2])
