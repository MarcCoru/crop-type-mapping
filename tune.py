import ray
import ray.tune as tune
import argparse
from utils.parse_rayresults import parse_experiment
from utils.raytrainer import RayTrainerDualOutputRNN, RayTrainerConv1D
import datetime
import os
import sys


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

def get_hyperparameter_search_space(experiment):
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
            num_rnn_layers=tune.grid_search([1,2,3,4]),
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
            num_rnn_layers=tune.grid_search([1]),
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
            hidden_dims=tune.grid_search([25]),
            learning_rate=tune.grid_search([1e-2]),
            num_layers=tune.grid_search([1]),
            dataset=args.dataset)

    else:
        raise ValueError("please insert valid experiment name for search space (either 'rnn' or 'conv1d')")

def tune_dataset(args):

    config = get_hyperparameter_search_space(args.experiment)

    if args.experiment == "rnn" or args.experiment == "test_rnn":
        tune_dataset_rnn(args, config)
    elif args.experiment == "conv1d" or args.experiment == "test_conv1d":
        tune_dataset_rnn(args, config)

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
                "config": config
            }
        },
        verbose=0)

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
                "config": config
            }
        },
        verbose=0)

def tune_mori_datasets(args):
    """
    Calls tune_dataset on each dataset listed in the datasetfile.

    :param args: argparse arguments forwarded further
    """
    datasets = [dataset.strip() for dataset in open(args.datasetfile, 'r').readlines()]
    resultsdir = os.path.join(os.getenv("HOME"), "ray_results", args.experiment)

    if args.skip_processed:
        processed_datasets = os.listdir(resultsdir)
        # remove all datasets that are present in the folder already
        datasets = list(set(datasets).symmetric_difference(processed_datasets))

    # start ray server
    ray.init(include_webui=False)

    for dataset in datasets:
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        args.dataset = dataset
        try:

            tune_dataset(args)

            top = parse_experiment(experimentpath=os.path.join(resultsdir, dataset),
                                   outcsv=os.path.join(resultsdir, dataset, "params.csv"))
            num_hidden, learning_rate, num_rnn_layers = top.iloc[0].name
            param_string = "num_hidden:{}, learning_rate:{}, num_rnn_layers:{}".format(*top.iloc[0].name)
            perf_string = "accuracy {:.2f} (+-{:.2f}) in {:.0f} folds".format(top.iloc[0].mean_accuracy,
                                                                              top.iloc[0].std_accuracy,
                                                                              top.iloc[0].nfolds)
            print("{time} finished tuning dataset {dataset} {perf_string}, {param_string}".format(time=time,
                                                                                                  dataset=dataset,
                                                                                                  perf_string=perf_string,
                                                                                                  param_string=param_string),
                  file=open(os.path.join(resultsdir, "datasets.log"), "a"))

        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print("error" + str(e))
            continue

if __name__=="__main__":

    # parse input arguments
    args = parse_args()
    tune_mori_datasets(args)

