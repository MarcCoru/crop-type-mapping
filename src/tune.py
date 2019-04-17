import sys
sys.path.append("models")

import ray.tune as tune
import argparse
import datetime
import os
import torch
from utils.trainer import Trainer
import ray.tune
from argparse import Namespace

from train import prepare_dataset, getModel


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
        '-w', '--workers', type=int, default=2, help='cpu workers')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.2,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument(
        '-r', '--local_dir', type=str, default=os.path.join(os.environ["HOME"],"ray_results"),
        help='ray local dir. defaults to $HOME/ray_results')
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
            epochs = 10,
            model = "rnn",
            dataset = "BavarianCrops",
            classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf",
            num_layers = tune.grid_search([1,2,3,4]),
            hidden_dims = tune.grid_search([2**6,2**7,2**8]),
            samplet=tune.grid_search([30,50,70]),
            bidirectional = True,
            dropout=tune.grid_search([.25,.50,.75]),
            train_on="train",
            test_on="valid",
            trainregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"],
            testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"],
            )

    if experiment == "transformer":

        return dict(
            epochs = 10,
            model = "transformer",
            dataset = "BavarianCrops",
            classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf",
            hidden_dims = tune.grid_search([2**7,2**8,2**6]),
            n_heads = tune.grid_search([2,4,6,8]),
            n_layers = tune.grid_search([8,4,2,1]),
            samplet=tune.grid_search([30,50,70]),
            bidirectional = True,
            dropout=tune.grid_search([.25,.50,.75]),
            train_on="train",
            test_on="valid",
            trainregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"],
            testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"],
            )

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

class RayTrainerRNN(ray.tune.Trainable):
    def _setup(self, config):

        self.epochs = config["epochs"]

        args = Namespace(**config)
        self.traindataloader, self.validdataloader = prepare_dataset(args)

        args.nclasses = self.traindataloader.dataset.nclasses
        args.seqlength = self.traindataloader.dataset.sequencelength
        args.input_dims = self.traindataloader.dataset.ndims

        self.model = getModel(args)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if "model" in config.keys():
            config.pop('model', None)
        trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **config)

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train five epochs and then infer once. to avoid overhead on these small datasets
        for i in range(self.epochs):
            self.trainer.train_epoch(epoch=None)

        return self.trainer.test_epoch(self.validdataloader, epoch=None)

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

if __name__=="__main__":
    if not ray.is_initialized():
        ray.init(include_webui=False)

    args = parse_args()

    config = get_hyperparameter_search_space(args.experiment)

    args_dict = vars(args)
    config = {**config, **args_dict}
    args = Namespace(**config)

    tune.run_experiments(
        {
            args.experiment: {
                "resources_per_trial": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 1,
                    'time_total_s':3600 if not args.smoke_test else 1,
                },
                "run": RayTrainerRNN,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        verbose=0,)

