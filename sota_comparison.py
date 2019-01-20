import ray
import ray.tune as tune
import argparse
from utils.raytrainer import RayTrainerDualOutputRNN, RayTrainerConv1D
import datetime
import os
import sys
from utils.rayresultsparser import RayResultsParser
from utils.UCR_Dataset import UCRDataset
from models.conv_shapelets import ConvShapeletModel
import torch
from utils.trainer import Trainer
from train import get_datasets_from_hyperparametercsv, readHyperparameterCSV
import pandas as pd
import logging

def main():
    # parse input arguments
    args = parse_args()
    run_experiment_on_datasets(args)


def parse_args():
    parser = argparse.ArgumentParser("e.g. execute: /data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv -b 16 -c 2 -g .25 --skip-processed -r /tmp")
    parser.add_argument(
        'hyperparametercsv', type=str, default="/data/remote/hyperparams_conv1d_v2/hyperparams.csv/hyperparams_conv1d.csv",
        help='csv containing hyper parameters')
    parser.add_argument(
        '-x', '--experiment', type=str, default="sota_comparison", help='Batch Size')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=96, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
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

def run_experiment(args):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    #experiment_name = args.dataset
    datasets = get_datasets_from_hyperparametercsv(args.hyperparametercsv)

    if args.experiment == "sota_comparison":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=30, # will be overwritten by training_iteration criterion
                switch_epoch=15,
                earliness_factor=tune.grid_search([0.6, 0.7, 0.8, 0.9]),
                entropy_factor=0.01,
                ptsepsilon=tune.grid_search([5,10,20]),
                hyperparametercsv=args.hyperparametercsv,
                dataset=tune.grid_search(datasets),
                drop_probability=0.5,
                lossmode="twophase_linear_loss",
            )
    if args.experiment == "entropy_pts":
        config = dict(
                batchsize=args.batchsize,
                workers=2,
                epochs=60, # will be overwritten by training_iteration criterion
                switch_epoch=30,
                earliness_factor=tune.grid_search([0.6, 0.7, 0.8, 0.9]),
                entropy_factor=tune.grid_search([0, 0.01, 0.1]),
                ptsepsilon=tune.grid_search([0, 5, 10]),
                hyperparametercsv=args.hyperparametercsv,
                dataset=tune.grid_search(datasets),
                drop_probability=0.5,
                lossmode="twophase_linear_loss",
            )

    tune.run_experiments(
        {
            args.experiment: {
                "trial_resources": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 1, # 1 iteration = 60 training epochs plus 1 eval epoch
                    'time_total_s':600 if not args.smoke_test else 1,
                },
                "run": RayTrainer,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        verbose=0)

def run_experiment_on_datasets(args):
    """
    Calls tune_dataset on each dataset listed in the datasetfile.

    :param args: argparse arguments forwarded further
    """
    rayresultparser = RayResultsParser()

    datasets = get_datasets_from_hyperparametercsv(args.hyperparametercsv)
    resultsdir = os.path.join(args.local_dir, "sota_comparison")
    args.local_dir = resultsdir

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    if args.skip_processed:
        processed_datasets = [f for f in os.listdir(resultsdir) if os.path.isdir(os.path.join(resultsdir,f))]
        print("--skip-processed option enabled. Found {}/{} datasets present. skipping these...".format(len(datasets),len(processed_datasets)))
        # remove all datasets that are present in the folder already
        datasets = list(set(datasets).symmetric_difference(processed_datasets))

    # start ray server
    if not ray.is_initialized():
        ray.init(include_webui=False, configure_logging=True, logging_level=logging.INFO)

    try:
        run_experiment(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print("error" + str(e))


class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):
        self.dataset = config["dataset"]
        self.earliness_factor = config["earliness_factor"]

        hparams = pd.read_csv(config["hyperparametercsv"]).set_index("dataset").loc[self.dataset]

        config["learning_rate"] = float(hparams.learning_rate)
        config["num_layers"] = int(hparams.num_layers)
        config["hidden_dims"] = int(hparams.hidden_dims)
        config["shapelet_width_increment"] = int(hparams.shapelet_width_increment)

        traindataset = UCRDataset(config["dataset"],
                                  partition="trainvalid",
                                  silent=True,
                                  augment_data_noise=0)

        validdataset = UCRDataset(config["dataset"],
                                  partition="test",
                                  silent=True)

        self.epochs = config["epochs"]

        nclasses = traindataset.nclasses

        # handles multitxhreaded batching andconfig shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
                                                           num_workers=config["workers"],
                                                           pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
                                                      num_workers=config["workers"], pin_memory=False)

        self.model = ConvShapeletModel(num_layers=config["num_layers"],
                                       hidden_dims=config["hidden_dims"],
                                       ts_dim=1,
                                       n_classes=nclasses,
                                       use_time_as_feature=True,
                                       drop_probability=config["drop_probability"],
                                       scaleshapeletsize=False,
                                       shapelet_width_increment=config["shapelet_width_increment"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train epochs and then infer once. to avoid overhead on these small datasets
        for epoch in range(self.epochs):
            self.trainer.epoch = epoch
            self.trainer.train_epoch(epoch=None)

        return self.trainer.test_epoch(epoch=None)

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

if __name__=="__main__":

    main()
