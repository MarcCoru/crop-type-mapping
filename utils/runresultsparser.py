import os
import pandas as pd

LOGFILE_PHASE1 = "log_classification.csv"
LOGFILE_PHASE2 = "log_earliness.csv"

def get_earliness_accuracy_last_run(file):
    run = pd.read_csv(file)
    accuracy = run.loc[run["mode"] == "test"].iloc[-1]["accuracy"]
    earliness = run.loc[run["mode"] == "test"].iloc[-1]["earliness"]
    return earliness, accuracy

def parse_run(root, outcsv=None):

    results = list()
    for dataset in os.listdir(root):
        datasetpath = os.path.join(root,dataset)

        try:
            ph1_earliness, ph1_accuracy = get_earliness_accuracy_last_run(os.path.join(datasetpath, LOGFILE_PHASE1))
            ph2_earliness, ph2_accuracy = get_earliness_accuracy_last_run(os.path.join(datasetpath, LOGFILE_PHASE2))

            summary = dict(
                dataset=dataset,
                phase1_earliness=ph1_earliness,
                phase1_accuracy=ph1_accuracy,
                phase2_earliness=ph2_earliness,
                phase2_accuracy=ph2_accuracy
            )

            results.append(summary)
        except Exception as e:
            print("Could not read dataset {}. skipping...".format(dataset))

    if len(results) > 0:
        dataframe=pd.DataFrame(results).set_index("dataset")
        if outcsv is not None:
            print("Writing runs to {}".format(outcsv))
            dataframe.to_csv(outcsv)

        return dataframe
    else:
        return None

if __name__=="__main__":
    root = "/data/remote/early_rnn/conv1d"
    outpath = "../viz/data/twophase_linear_loss"

    os.makedirs(outpath,exist_ok=True)

    for run in os.listdir(root):
        if os.path.isdir(os.path.join(root,run)):
            parse_run(os.path.join(root,run), outcsv=os.path.join(outpath,run+".csv"))

