import pandas as pd

loss = "twophase_linear_loss"
alpha=0.6
entropy_factor=0.01

relclass_t = "t=0.001"
edsc_t = "t=2.5"
ects_col = "sup=0.05"

mori = pd.read_csv("data/morietal2017/mori-accuracy-sr2-cf2.csv", sep=' ').set_index("Dataset")
mori=mori["a={}".format(alpha)]
mori.name = "morietal2017"

relclass = pd.read_csv("data/morietal2017/relclass-accuracy-gaussian-quadratic-set.csv", sep=' ').set_index("Dataset")
relclass=relclass[relclass_t]
relclass.name = "relclass"

edsc = pd.read_csv("data/morietal2017/edsc-accuracy.csv", sep=' ').set_index("Dataset")
edsc=edsc[edsc_t]
edsc.name = "edsc"

etsc = pd.read_csv("data/morietal2017/ects-accuracy-strict-method.csv", sep=' ').set_index("Dataset")
etsc=etsc[ects_col]
etsc.name = "etsc"


csvfile = "data/{loss}/a{alpha}e{entropy_factor}.csv".format(loss=loss, alpha=alpha, entropy_factor=entropy_factor)

ours = pd.read_csv(csvfile, index_col=0)["phase2_accuracy"]*100
ours.name = "ours"

concated = pd.concat([etsc,edsc,relclass,mori, ours], axis=1, join='inner')

concated<concated["ours"]

pass