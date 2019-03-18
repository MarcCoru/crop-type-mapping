import ray
import ray.tune as tune
import argparse
import datetime
import os
import sys
from utils.rayresultsparser import RayResultsParser
from models.DualOutputRNN import DualOutputRNN
from models.ConvShapeletModel import ConvShapeletModel
from models.rnn import RNN
from datasets.UCR_Dataset import UCRDataset
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
from utils.trainer import Trainer
import ray.tune


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
        '-r', '--local_dir', type=str, default=os.path.join(os.environ["HOME"],"ray_results"),
        help='ray local dir. defaults to $HOME/ray_results')
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
            epochs=30,
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([0]), #[0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
            learning_rate=tune.grid_search([1e-2,1e-3,1e-4]),
            dropout=0.5,
            num_layers=tune.grid_search([1,2,3,4]),
            dataset=args.dataset)

    if experiment == "test_rnn":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=1,
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([5]), #[0, 1, 2, 3, 4]),
            hidden_dims=tune.grid_search([2 ** 6]),
            learning_rate=tune.grid_search([1e-2]),
            dropout=0.3,
            num_layers=tune.grid_search([1,2]),
            dataset=args.dataset)

    elif experiment == "conv1d":

        #initial search space for conv1d (Jan 2nd 2019)
        #return dict(
        #    batchsize=args.batchsize,
        #    workers=2,
        #    epochs=99999, # will be overwritten by training_iteration criterion
        #    switch_epoch=9999,
        #    earliness_factor=1,
        #    fold=tune.grid_search([0,1,2,3,4]),
        #    hidden_dims=tune.grid_search([10,25,50,75]),
        #    learning_rate=tune.grid_search([1e-1,1e-2,1e-3,1e-4]),
        #    num_layers=tune.grid_search([2,3,4]),
        #    dataset=args.dataset)

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=15,  # pure train epochs. then one validation...
            switch_epoch=9999,
            earliness_factor=1,
            hidden_dims=tune.grid_search([50, 75, 100]),
            num_layers=tune.grid_search([8,6,4,2]),
            drop_probability=tune.grid_search([0.25, 0.5, 0.75]),
            shapelet_width_increment=tune.grid_search([30, 50, 70]),
            learning_rate=tune.grid_search([1e-1, 1e-2]),
            fold=tune.grid_search([0, 1, 2]),
            dataset=args.dataset)

    elif experiment == "test_conv1d":

        return dict(
            batchsize=args.batchsize,
            workers=2,
            epochs=1, # will be overwritten by training_iteration criterion
            switch_epoch=9999,
            earliness_factor=1,
            fold=tune.grid_search([0]),
            hidden_dims=tune.grid_search([25,50]),
            learning_rate=tune.grid_search([1e-2]),
            num_layers=tune.grid_search([1]),
            dataset=args.dataset,
            drop_probability=0.5,
            shapelet_width_increment=tune.grid_search([10]),
        )

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
                    'training_iteration': 1,
                    'time_total_s':600 if not args.smoke_test else 1,
                },
                "run": RayTrainerRNN,
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
                    'training_iteration': 1, # 1 iteration = 60 training epochs plus 1 eval epoch
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
    rayresultparser = RayResultsParser()

    datasets = [dataset.strip() for dataset in open(args.datasetfile, 'r').readlines()]
    resultsdir = os.path.join(args.local_dir, args.experiment)
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
        ray.init(include_webui=False)

    for dataset in datasets:
        args.dataset = dataset
        try:

            tune_dataset(args)

            experimentpath = os.path.join(resultsdir, dataset)
            if not os.path.exists(experimentpath):
                os.makedirs(experimentpath)

            if args.experiment == "test_conv1d" or args.experiment == "conv1d":
                searched_parameters = ["hidden_dims", "learning_rate", "num_layers", "shapelet_width_increment"]
            elif args.experiment == "test_rnn" or args.experiment == "rnn":
                searched_parameters=["hidden_dims", "learning_rate", "num_layers"]
            rayresultparser.get_best_hyperparameters(resultsdir, hyperparametercsv=resultsdir+"/hyperparameter.csv",
                                                     group_by=searched_parameters)

            top = rayresultparser._get_n_best_runs(experimentpath=experimentpath,n=1,group_by=searched_parameters)
            print_best(top, filename=os.path.join(resultsdir, "datasets.log"))


        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            #print("error" + str(e))
            #continue
            raise

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

        region = "HOLL_2018_MT_pilot"
        root = "/home/marc/data/BavarianCrops"
        nsamples=None
        classmapping = "/home/marc/data/BavarianCrops/classmapping.csv.gaf"
        traindataset = BavarianCropsDataset(root=root, region=region, partition="train", nsamples=None, classmapping=classmapping)
        validdataset = BavarianCropsDataset(root=root, region=region, partition="valid", nsamples=None,
                                            classmapping=classmapping)


        nclasses = traindataset.nclasses

        self.epochs = config["epochs"]

        # handles multitxhreaded batching andconfig shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
                                                           num_workers=config["workers"],
                                                           pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
                                                      num_workers=config["workers"], pin_memory=False)

        self.model = RNN(input_dim=traindataset.ndims,
                                   nclasses=nclasses,
                                   hidden_dims=config["hidden_dims"],
                                   num_rnn_layers=config["num_layers"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

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

    # parse input arguments
    args = parse_args()
    args.dataset = "BavarianCrops"
    tune_dataset(args)

