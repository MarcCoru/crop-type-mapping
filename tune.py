import ray
import ray.tune as tune
import argparse
#from utils.parse_rayresults import parse_experiment
from utils.raytrainer import RayTrainerDualOutputRNN, RayTrainerConv1D
import datetime
import os
import sys
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment', type=str, default="rnn",
        help='experiment name. defines hyperparameter search space and tune dataset function'
             "use 'rnn', 'test_rnn', 'conv1d', or 'test_conv1d'")
    parser.add_argument(
        '-d', '--datasetfile', type=str, default="experiments/morietal2017/UCR_dataset_names.txt",
            help='text file containing dataset names in new lines')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=96, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.2,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument(
        '-r', '--local_dir', type=str, default="~/ray_results",
        help='ray local dir. defaults to ~/ray_results')
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    parser.add_argument(
        '--skip-processed', action='store_true', help='skip already processed datasets (defined by presence of results folder)')
    args, _ = parser.parse_known_args()
    return args

def get_hyperparameter_search_space(experiment, args):
    """
    simple state function to hold the parameter search space definitions for experiments

    :param experiment: experiment name
    :return: ray config dictionary
    """
    if experiment == "rnn":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=99999,
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([5,6,7,8,9]), #[0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
            learning_rate=tune.grid_search([1e-2,1e-3,1e-4]),
            dropout=0.3,
            num_layers=tune.grid_search([1,2,3,4]),
            dataset=args.dataset)

    if experiment == "test_rnn":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=99999,
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([5]), #[0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([2 ** 6]),
            learning_rate=tune.grid_search([1e-2]),
            dropout=0.3,
            num_layers=tune.grid_search([1,2]),
            dataset=args.dataset)

    elif experiment == "conv1d":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=99999, # will be overwritten by training_iteration criterion
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([25,50,75]),
            learning_rate=tune.grid_search([1e-2,1e-3,1e-4]),
            num_layers=tune.grid_search([1,2,3,4]),
            dataset=args.dataset)

    elif experiment == "test_conv1d":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=99999, # will be overwritten by training_iteration criterion
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([0]),
            hidden_dims=tune.grid_search([25,50]),
            learning_rate=tune.grid_search([1e-2]),
            num_layers=tune.grid_search([1]),
            dataset=args.dataset)

    else:
        raise ValueError("please insert valid experiment name for search space (either 'rnn' or 'conv1d')")

def tune_dataset(args):

    config = get_hyperparameter_search_space(args.experiment, args)

    if args.experiment == "rnn" or args.experiment == "test_rnn":
        tune_dataset_rnn(args, config)
    elif args.experiment == "conv1d" or args.experiment == "test_conv1d":
        tune_dataset_conv1d(args, config)

def tune_dataset_rnn(args, config):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    experiment_name = args.dataset

    tune.run_experiments(
        {
            experiment_name: {
                "trial_resources": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 10 if not args.smoke_test else 1,
                    'time_total_s':600 if not args.smoke_test else 1,
                },
                "run": RayTrainerDualOutputRNN,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        verbose=0,)

def tune_dataset_conv1d(args, config):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    experiment_name = args.dataset

    tune.run_experiments(
        {
            experiment_name: {
                "trial_resources": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 10 if not args.smoke_test else 1,
                    'time_total_s':600 if not args.smoke_test else 1,
                },
                "run": RayTrainerConv1D,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        verbose=0)

def tune_mori_datasets(args):
    """
    Calls tune_dataset on each dataset listed in the datasetfile.

    :param args: argparse arguments forwarded further
    """
    datasets = [dataset.strip() for dataset in open(args.datasetfile, 'r').readlines()]
    resultsdir = os.path.join(args.local_dir, args.experiment)
    args.local_dir = resultsdir

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    if args.skip_processed:
        processed_datasets = os.listdir(resultsdir)
        # remove all datasets that are present in the folder already
        datasets = list(set(datasets).symmetric_difference(processed_datasets))

    # start ray server
    if not ray.is_initialized():
        ray.init(include_webui=False)

    for dataset in datasets:
        args.dataset = dataset
        try:

            tune_dataset(args)

            experimentpath = os.path.join(resultsdir, dataset)
            if not os.path.exists(experimentpath):
                os.makedirs(experimentpath)

            top = parse_experiment(experimentpath=experimentpath,
                                   outcsv=os.path.join(experimentpath, "params.csv"))

            print_best(top,filename=os.path.join(resultsdir, "datasets.log"))

        except KeyboardInterrupt:
            sys.exit(0)
        #except Exception as e:
        #    print("error" + str(e))
        #    continue

def print_best(top, filename):
    """
    Takes best run from pandas dataframe <top> and writes parameter and accuracy info to a text file
    """
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # num_hidden, learning_rate, num_rnn_layers = top.iloc[0].name
    best_run = top.iloc[0]

    param_fmt = "hidden_dims:{hidden_dims}, learning_rate:{learning_rate}, num_layers:{num_layers}"
    param_string = param_fmt.format(hidden_dims=best_run.loc["hidden_dims"],
                                    learning_rate=best_run.loc["learning_rate"],
                                    num_layers=best_run["num_layers"])

    performance_fmt = "accuracy {accuracy:.2f} (+-{std:.2f}) in {folds:.0f} folds"
    perf_string = performance_fmt.format(accuracy=best_run.mean_accuracy,
                                         std=best_run.std_accuracy,
                                         folds=best_run.nfolds)

    print("{time} finished tuning dataset {dataset} {perf_string}, {param_string}".format(time=time,
                                                                                          dataset=best_run.dataset,
                                                                                          perf_string=perf_string,
                                                                                          param_string=param_string),
          file=open(filename, "a"))


def load_run(path):

    result_file = os.path.join(path, "result.json")

    if not os.path.exists(result_file):
        return None

    with open(result_file,'r') as f:
        lines = f.readlines()

    if len(lines) > 0:
        result = json.loads(lines[-1])
        return result["accuracy"], result["loss"], result["training_iteration"], result["timestamp"], result["config"]
    else:
        return None

def load_experiment(path):
    runs = os.listdir(path)

    result = list()
    for run in runs:
        runpath = os.path.join(path, run)

        run = load_run(runpath)
        if run is None:
            continue
        else:
            accuracy, loss, training_iteration, timestamp, config = run

        result.append(
            dict(
                accuracy=accuracy,
                loss=loss,
                training_iteration=training_iteration,
                batchsize=config["batchsize"],
                dataset=config["dataset"],
                hidden_dims=config["hidden_dims"],
                num_layers=config["num_layers"],
                learning_rate=config["learning_rate"],
                fold=config["fold"],
                dropout=config["dropout"] if "dropout" in config.keys() else None
            )
        )

    return result

def parse_experiment(experimentpath, outcsv=None, n=5):
    result = load_experiment(experimentpath)

    if len(result) == 0:
        print("Warning! Experiment {} returned no runs".format(experimentpath))
        return None

    result = pd.DataFrame(result)
    # average accuracy over the same columns (particularily over the fold variable...)
    grouped = result.groupby(["hidden_dims", "learning_rate", "num_layers"])["accuracy"]
    nfolds = grouped.count().rename("nfolds")
    mean_accuracy = grouped.mean().rename("mean_accuracy")
    std_accuracy = grouped.std().rename("std_accuracy")

    score = pd.concat([mean_accuracy, std_accuracy, nfolds], axis=1)

    top = score.nlargest(n, "mean_accuracy")

    dataset = os.path.basename(experimentpath)
    top.reset_index(inplace=True)
    top["dataset"] = dataset

    if outcsv is not None:
        top.to_csv(outcsv)

    return top

def load_set_of_experiments(path):
    experiments = os.listdir(path)

    best_hyperparams = list()
    for experiment in experiments:

        experimentpath = os.path.join(path,experiment)

        if os.path.isdir(experimentpath):
            print("parsing experiment "+experiment)
            result = parse_experiment(experimentpath=experimentpath, outcsv=None, n=1)
            if result is not None:
                best_hyperparams.append(result)

    summary = pd.concat(best_hyperparams)

    csvfile = os.path.join(path, "hyperparams.csv")
    print("writing "+csvfile)
    summary.to_csv(csvfile)

if __name__=="__main__":

    # parse input arguments
    args = parse_args()
    tune_mori_datasets(args)

