import geopandas as gpd
import numpy as np
import os
import pandas as pd
from shutil import copyfile
import sys

run = sys.argv[1]

#outpath = "/tmp/eval"
outpath = sys.argv[2]

#classmapping = "/home/marc/data/BavarianCrops/classmapping.csv.gaf"
classmapping = sys.argv[3]


#run="/home/marc/projects/gafreport/images/data/TUM_ALL_rnn"
epoch=35

os.makedirs(outpath,exist_ok=True)

path = "/home/marc/data/BavarianCrops/shp/"
regions = ["NOWA_2018_MT_pilot.shp","HOLL_2018_MT_pilot.shp","KRUM_2018_MT_pilot.shp"]

outshp = os.path.join(outpath,"eval.shp")


print("copying {} to {}".format(classmapping, os.path.join(outpath,"classmapping.csv")))
copyfile(classmapping, os.path.join(outpath,"classmapping.csv"))

mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
#mapping = mapping.set_index("nutzcode")
classes = mapping["id"].unique()
classname = mapping.groupby("id").first().classname.values
#nutzcodes = mapping.groupby("id").first().nutzcode.values
klassenname = mapping.groupby("id").first().klassenname.values

#code2id = dict(zip(nutzcodes,classes))
onehot = np.eye(len(classes)).astype(bool)

os.makedirs(os.path.dirname(outshp),exist_ok=True)


ids = np.load(run+"/npy/ids_{}.npy".format(epoch))

probas = np.load(run+"/npy/probas_{}.npy".format(epoch))
probas_df = pd.DataFrame(probas, columns=["prob_"+str(cl) for cl in np.arange(probas.shape[1])], index=ids)

probas_df["pred"] = probas.argsort()[:,-1]
assert (probas.argsort()[:,-1] == probas.argmax(1)).all()
probas_df["pred2nd"] = probas.argsort()[:,-2]

probas_df["prob2nd"] = probas[onehot[probas_df["pred2nd"].values]]
probas_df["maxprob"] = probas[onehot[probas_df["pred"].values]]


targets = np.load(run+"/npy/labels_{}.npy".format(epoch))
probas_df["GRPGRPSTM"] = targets
probas_df["correct_prediction"] = probas_df["GRPGRPSTM"] == probas_df["pred"]

probas_df["score_correct"] = probas[onehot[probas_df["GRPGRPSTM"].values]]

shp_path=[os.path.join(path,f) for f in regions]

shps = list()
for shp_p in shp_path:
    print("loading "+shp_p)
    shps.append(gpd.read_file(shp_p).set_index("ID"))
shps = pd.concat(shps)

print("merging predictions with shapefile")
shps = shps.join(probas_df, how='inner')

shps["pred_name"] = klassenname[shps["pred"]]
shps["corr_name"] = klassenname[shps["GRPGRPSTM"]]

print("writing "+outshp)
shps.to_file(outshp, encoding="utf-8")
pass