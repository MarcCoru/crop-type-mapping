import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


run = "/data/EV2019/early_reward"
states = os.path.join(run,"BavarianCrops","npy")
target = "/home/marc/projects/EV2019/images/example"

pass

file = "{array}_{epoch}.npy"
epoch = 100
sample = -1

inputs = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="inputs",epoch=epoch)))
probas = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="probas",epoch=epoch)))
tstops = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="t_stops",epoch=epoch)))
pts = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="pts",epoch=epoch)))
deltas = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="deltas",epoch=epoch)))
budgets = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="budget",epoch=epoch)))
targets = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="targets",epoch=epoch)))
predictions = np.load(os.path.join(run,"BavarianCrops","npy",file.format(array="predictions",epoch=epoch)))

T = inputs.shape[1]

input = inputs[sample]
proba = probas[:,sample,:]
tstop = tstops[sample] / T # normalize from idx 0-70 to float 0-1
pt = pts[sample]
delta = deltas[sample]
budget = budgets[sample]
label = targets[sample,0]
prediction =predictions[sample]

fix, axs = plt.subplots(5,1)

axs[0].plot(input)
axs[0].axvline(x=tstop*T)

axs[1].plot(proba)
axs[1].axvline(x=tstop*T)

axs[2].bar(np.arange(len(pt)), pt)
axs[2].axvline(x=tstop*T)

axs[3].bar(np.arange(len(delta)), delta)
axs[3].axvline(x=tstop*T)

axs[4].bar(np.arange(len(budget)), budget)
axs[4].axvline(x=tstop*T)




def write(input, proba, tstop, label, prediction):
    bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    print("writing "+os.path.join(target,"input.csv"))

    df = pd.DataFrame(input, columns=bands).reset_index()
    df["t"] = df["index"] / len(df)
    df.to_csv(os.path.join(target,"input.csv"))

    classnames = ["meadows","winter barley","corn","winter wheat","summer barley","clover","winter triticale"]
    print("writing " + os.path.join(target, "proba.csv"))

    df = pd.DataFrame(proba, columns=classnames).reset_index()
    df["t"] = df["index"] / len(df)
    df.to_csv(os.path.join(target,"proba.csv"))

    print("writing " + os.path.join(target, "tstop.txt"))
    with open(os.path.join(target,"tstop.txt"),"w") as f:
        f.write("tstop, label, prediction\n")
        f.write("{}, {}, {}".format(tstop, classnames[label], classnames[prediction]))

write(input, proba, tstop, label, prediction)


# loss illustration

t_index = 1 - np.arange(T)/T

yplus = probas[:,sample, label]

fix, axs = plt.subplots(3,1)
axs[0].plot(yplus)
axs[1].bar(np.arange(len(t_index)), t_index)
axs[2].bar(np.arange(len(pt)), pt)


fix, axs = plt.subplots(2,1)
axs[0].plot(-np.log(yplus))
axs[1].bar(np.arange(len(pt)), pt)


df = pd.DataFrame([yplus,t_index,pt,-np.log(yplus)]).T
df.columns = ["yplus","t_index","pt","logyplus"]
print("writing /home/marc/projects/EV2019/images/lossexample.csv")
df.to_csv("/home/marc/projects/EV2019/images/lossexample.csv")
plt.show()
pass

