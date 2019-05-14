import geopandas as gpd
import numpy as np
import os
import pandas as pd
from shutil import copyfile

run="/home/marc/projects/gafreport/images/data/TUM_ALL_rnn"
epoch=35
outpath = "/tmp/eval"
os.makedirs(outpath,exist_ok=True)

path = "/home/marc/data/BavarianCrops/shp/"
regions = ["NOWA_2018_MT_pilot.shp","HOLL_2018_MT_pilot.shp","KRUM_2018_MT_pilot.shp"]

outshp = os.path.join(outpath,"eval.shp")
classmapping = "/home/marc/data/BavarianCrops/classmapping.csv.gaf"

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

probas = np.load(run+"/npy/probas_{}.npy".format(epoch))
ids = np.load(run+"/npy/ids_{}.npy".format(epoch))
targets = np.load(run+"/npy/labels_{}.npy".format(epoch))

shp_path=[os.path.join(path,f) for f in regions]

shps = list()
for shp_p in shp_path:
    print("loading "+shp_p)
    shps.append(gpd.read_file(shp_p).set_index("ID"))
shps = pd.concat(shps)

result = pd.DataFrame(probas, columns=["prob_"+str(cl) for cl in np.arange(probas.shape[1])])
result.index = ids

print("merging predictions with shapefile")
shps = shps.join(result, how='inner')

predicted = probas.argmax(1)
predicted_name = klassenname[probas.argmax(1)]
score_predicted = probas[onehot[predicted]]
score_targets = probas[onehot[targets]]
target_name = klassenname[targets]

shps["pred"] = predicted
shps["GRPGRPSTM"] = targets
shps["pred_name"] = predicted_name
shps["corr_name"] = target_name
shps["maxprob"] = score_predicted
shps["score_target"] = score_targets
shps["correct_prediction"] = targets == predicted

print("writing "+outshp)
shps.to_file(outshp, encoding="utf-8")
pass