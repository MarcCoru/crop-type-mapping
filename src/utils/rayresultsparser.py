import os
import json
import pandas as pd

class RayResultsParser():
    def __init__(self):
        pass

    def _load_run(self, path):

        result_file = os.path.join(path, "result.json")
        #result_file = path

        if not os.path.exists(result_file):
            return None

        with open(result_file,'r') as f:
            lines = f.readlines()

        if len(lines) > 0:
            return json.loads(lines[-1])
        else:
            return None

    def _load_all_runs(self, path):
        runs = [r for r in os.listdir(path) if os.path.isdir(os.path.join(path,r))]

        result_list = list()
        for run in runs:
            runpath = os.path.join(path, run)

            run = self._load_run(runpath)
            if run is None:
                continue
            else:
                result = run

            config = result.pop("config")

            # merge result dict and config dict
            for key, value in config.items():
                result[key] = value

            result_list.append(result)

        return result_list

    def _get_n_best_runs(self,
                         experimentpath,
                         n=5,
                         group_by=["hidden_dims", "learning_rate", "num_rnn_layers"]):

        resultlist = self._load_all_runs(experimentpath)

        if len(resultlist) == 0:
            print("Warning! Experiment {} returned no runs".format(experimentpath))
            return None

        result = pd.DataFrame(resultlist)
        # average accuracy over the same columns (particularily over the fold variable...)
        grouped = result.groupby(group_by)["accuracy"]

        nfolds = grouped.count().rename("nfolds")
        mean_accuracy = grouped.mean().rename("mean_accuracy")
        std_accuracy = grouped.std().rename("std_accuracy")

        score = pd.concat([mean_accuracy, std_accuracy, nfolds], axis=1)

        top = score.nlargest(n, "mean_accuracy")
        top["runs"] = len(score)

        dataset = os.path.basename(experimentpath)
        top.reset_index(inplace=True)
        top["dataset"] = dataset

        return top

    def get_sota_experiment(self, path, outpath=None, columns=["accuracy", "earliness"]):
        data = self._load_all_runs(path)
        print("{} runs returned!".format(len(data)))
        data = pd.DataFrame(data).set_index(["dataset"])
        data.sort_values(by="accuracy", ascending=False).drop_duplicates(
            subset=['earliness_factor', 'entropy_factor', 'ptsepsilon'], keep='first')
        data[columns].to_csv(outpath)

    def get_best_hyperparameters(self, path, hyperparametercsv=None, group_by=["hidden_dims", "learning_rate", "num_rnn_layers"], n=1):

        experiments = os.listdir(path)

        best_hyperparams = list()
        for experiment in experiments:

            experimentpath = os.path.join(path,experiment)

            if os.path.isdir(experimentpath):
                print("parsing experiment "+experiment)
                result = self._get_n_best_runs(experimentpath=experimentpath, n=n, group_by=group_by)
                if result is not None:
                    best_hyperparams.append(result)

        summary = pd.concat(best_hyperparams)

        if hyperparametercsv is not None:
            outpath = os.path.dirname(hyperparametercsv)

            if not os.path.exists(outpath):
                os.makedirs(outpath,exist_ok=True)

            print("writing "+hyperparametercsv)
            summary.to_csv(hyperparametercsv)

        return summary

def parse_hyperparameters(rayroot="/data/remote/hyperparams_conv1d_v2_secondrun/conv1d",
                          outcsv="/data/remote/hyperparams_conv1d_v2_secondrun/hyperparams.csv"):
    parser = RayResultsParser()
    summary = parser.get_best_hyperparameters(rayroot,
                                              hyperparametercsv=outcsv,
                                              group_by=["hidden_dims", "learning_rate", "num_layers",
                                                        "shapelet_width_increment"])

    print(summary.set_index("dataset")[["mean_accuracy", "std_accuracy", "runs"]])

def parse_sota_experiment(path, outcsv):
    parser = RayResultsParser()

    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    print("writing to "+outcsv)
    parser.get_sota_experiment(path,
                               outpath=outcsv,columns=["earliness_factor","entropy_factor","ptsepsilon","accuracy","earliness","lossmode"])

def parse_entropy_experiment():

    parser = RayResultsParser()

    outpath = "../viz/data/entropy_pts/runs.csv"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    print("writing to "+outpath)
    parser.get_sota_experiment("/data/remote/entropy_pts/entropy_pts",
                               outpath=outpath, columns=["earliness_factor","entropy_factor","accuracy","earliness"])

def save_tex(df, outfile):

    df["acc"] *= 100

    df = df.set_index("acc")
    df.index = df.index.astype(int)

    tex = df.to_latex(float_format=lambda x: '%10.0f' % x, escape=False, na_rep='')

    print("writing latex tabular to " + outfile)
    print(tex, file=open(outfile, "w"))

if __name__=="__main__":

    # hyperparameters train on train partition evaluated on validation partition in multiple folds
    #parse_hyperparameters(rayroot="/home/marc/ray_results/crops/",
    #                      outcsv="/home/marc/ray_results/crops/hyperparams.csv")

    parser = RayResultsParser()
    #summary = parser.get_best_hyperparameters("/data/remote/croptypemapping/rnn",
    #                                          hyperparametercsv="/data/remote/croptypemapping/rnn.csv",
    #                                          group_by=["hidden_dims", "learning_rate", "num_layers", "dropout"], n=5)


    result = pd.DataFrame(parser._load_all_runs("/data/remote/croptypemapping/rnn/"))

    top = result.sort_values(by="accuracy", ascending=False).iloc[:20]
    top = top[["accuracy","hidden_dims", "num_layers", "dropout", "samplet"]]
    top["dropout"] *= 100
    top["dropout"].astype(int)
    top.columns = ["acc", "$h$", "$L$", "$d$", "$T$"]
    save_tex(top, "/home/marc/projects/gafreport/images/hyperparam/rnn.csv")

    result = pd.DataFrame(parser._load_all_runs("/data/remote/croptypemapping/transformer/"))
    top = result.sort_values(by="accuracy", ascending=False).iloc[:20]
    top["dropout"] *= 100
    top["dropout"].astype(int)
    top = top[["accuracy","hidden_dims", "n_layers", "samplet", "dropout","n_heads"]]
    top.columns = ["acc", "$h$", "$L$", "$T$", "$d$", "$H$"]
    save_tex(top,outfile="/home/marc/projects/gafreport/images/hyperparam/transformer.csv")



    #print(summary.set_index("dataset")[["mean_accuracy", "std_accuracy", "runs"]])

    # results from runs using hyperparameters trained on train+valid partitions and tested on test partition
    #parse_sota_experiment(path="/data/remote/early_rnn/sota_comparison",
    #                      outcsv="../viz/data/sota_comparison/runs.csv")
