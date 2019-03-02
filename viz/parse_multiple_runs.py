import pandas as pd
import os

root = "/data/EV2019"
csv = "/home/marc/projects/EV2019/images/logs/data/early_reward/alphas.csv"

runs = [
    "earlyrewarda0",
    "earlyrewarda0.2",
    "earlyrewarda0.4",
    "earlyrewarda0.6",
    "earlyrewarda0.8",
    "earlyrewarda1"
]

runpaths = [os.path.join(root,run) for run in runs]

last_results = list()
for runpath in runpaths:
    data = pd.read_csv(os.path.join(runpath,"BavarianCrops","log_earliness.csv"))
    series = (data.loc[data["mode"] == "test"].set_index("epoch").loc[100])
    series["alpha"] = float(os.path.basename(runpath).replace("earlyrewarda",""))
    os.path.basename(runpath)
    last_results.append(series)

df = pd.DataFrame(last_results).set_index("alpha")

print("writing "+csv)
df.to_csv(csv)

print(r"alpha & accuracy & earliness & f1 & precision & recall & kappa \\")
for index, row in df.iterrows():

    print(r"{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\".format(index, row["accuracy"], row["earliness"], row["mean_f1"], row["mean_precision"], row["mean_recall"], row["kappa"]))

pass