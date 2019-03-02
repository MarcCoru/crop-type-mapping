import numpy as np
import os


root = "/data/EV2019/early_reward/BavarianCrops/"
target = "/home/marc/projects/EV2019/images/logs/data/early_reward/classes"
arraypath = os.path.join(root,"npy")
arrays = os.listdir(arraypath)

file = "{array}_{epoch}.npy"
os.path.exists(os.path.join(arraypath,file.format(array="t_stops", epoch=1)))

TMAX = 69

def load_class_tstop(epoch):

    t_stops = np.load(os.path.join(arraypath,file.format(array="t_stops", epoch=epoch))) / TMAX
    labels = np.load(os.path.join(arraypath,file.format(array="labels", epoch=epoch)))

    grouped = [t_stops[labels == i] for i in np.unique(labels)]
    return grouped

grouped = load_class_tstop(100)
classnames = ["meadows","winter barley","corn","winter wheat","summer barley","clover","winter triticale"]


# produce csv file for boxplots of the format:
#index median box_top box_bottom whisker_top whisker_bottom
# https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
print()
print("Boxplots metrics")
for cl in range(len(grouped)):
    data = grouped[cl]
    median = np.median(data)
    Q3 = np.percentile(data, 75)
    Q1 = np.percentile(data, 25)
    IQR = Q3-Q1
    max = Q3 + 1.5 * IQR
    min = Q1 - 1.5 * IQR

    print(cl, median, Q3, Q1, max, min, classnames[cl], sep=",")


pass
means = list()
medians = list()
stds = list()

for epoch in range(1,100):
    grouped = load_class_tstop(epoch)
    means.append([epoch] + [stop.mean() for stop in grouped])
    medians.append([epoch] + [np.median(stop) for stop in grouped])
    stds.append([epoch] + [np.std(stop) for stop in grouped])

import pandas as pd
means = pd.DataFrame(means, columns=["epoch"]+classnames).set_index("epoch")
medians = pd.DataFrame(medians, columns=["epoch"]+classnames).set_index("epoch")
stds = pd.DataFrame(stds, columns=["epoch"]+classnames).set_index("epoch")

means.to_csv(os.path.join(target,"mean.csv"))
medians.to_csv(os.path.join(target,"median.csv"))
stds.to_csv(os.path.join(target,"stds.csv"))
(means - stds).to_csv(os.path.join(target,"mean-std.csv"))
(means + stds).to_csv(os.path.join(target,"mean+std.csv"))




#pd.DataFrame(data, columns=["epoch","mean","median","std"])
#data.append(dict(
#    epoch=epoch,
#    mean=mean,
#    median=median,
#    std=std
#))