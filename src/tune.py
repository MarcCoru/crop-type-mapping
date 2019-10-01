import ray.tune as tune
import argparse
import datetime
import os
import torch
from utils.trainer import Trainer
import ray.tune
from argparse import Namespace
import torch.optim as optim
import numpy as np

from train import prepare_dataset, getModel

from config import HYPERBAND_BRACKETS, HYPERBAND_GRACE_PERIOD, RAY_EPOCHS, HYPERBAND_REDUCTION_FACTOR, CLASSMAPPING, RAY_NUM_SAMPLES, RAY_TEST_EVERY

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from utils.scheduled_optimizer import ScheduledOptim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment', type=str, default="rnn",
        help='experiment name. defines hyperparameter search space and tune dataset function'
             "use 'rnn', 'test_rnn', 'conv1d', or 'test_conv1d'")
    parser.add_argument(
        '-b', '--batchsize', type=int, default=96, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='cpu workers')
    parser.add_argument(
        '-m', '--max-concurrent', type=int, default=4, help='max concurrent runs')
    parser.add_argument(
        '--seed', type=int, default=None, help='random seed defaults to None')
    parser.add_argument(
        '--redis-address', type=str, default=None, help='address of ray tune head node: e.g. "localhost:6379"')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.2,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '-r', '--local_dir', type=str, default=os.path.join(os.environ["HOME"],"ray_results"),
        help='ray local dir. defaults to $HOME/ray_results')
    args, _ = parser.parse_known_args()
    return args

BavarianCrops_parameters = Namespace(
    dataset = "BavarianCrops",
    classmapping = CLASSMAPPING,
    samplet=70,
    scheme="blocks",
    train_on="train",
    test_on="valid",
    trainregions = ["holl", "nowa", "krum"],
    testregions = ["holl", "nowa", "krum"],
)

GAF_parameters = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl", "krum", "nowa"],
    testregions = ["holl", "krum", "nowa"],
    classmapping = CLASSMAPPING,
    scheme="blocks",
    features="optical",
    samplet=23,
    overwrite_cache=True,
    train_on="train",
    test_on="valid"
)

rnn_parameters = Namespace(
    model="rnn",
    num_layers=hp.choice("num_layers", [1, 2, 3, 4, 5, 6, 7]),
    hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8]),
    dropout=hp.uniform("dropout", 0, 1),
    weight_decay=hp.loguniform("weight_decay", -1, -12),
    learning_rate=hp.loguniform("learning_rate", -1, -8),
    bidirectional=True,
)

msresnet_parameters = Namespace(
    model="msresnet",
    hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8]),
    weight_decay=hp.loguniform("weight_decay", -1, -12),
    learning_rate=hp.loguniform("learning_rate", -1, -8)
)

tempCNN_parameters = Namespace(
    model="tempcnn",
    kernel_size=hp.choice("kernel_size", [3,5,7]),
    hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8]),
    dropout=hp.uniform("dropout", 0, 1),
    weight_decay=hp.loguniform("weight_decay", -1, -12),
    learning_rate=hp.loguniform("learning_rate", -1, -8)
)

transformer_parameters = Namespace(
    model = "transformer",
    hidden_dims = hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8]),
    n_heads = hp.choice("n_heads", [1,2,3,4,5,6,7,8]),
    n_layers = hp.choice("n_layers", [1, 2, 3, 4, 5, 6, 7, 8]),
    weight_decay = hp.loguniform("weight_decay", -1, -12),
    learning_rate = hp.loguniform("learning_rate", -1, -8),
    warmup = hp.choice("warmup", [1,10,100,1000]),
    dropout=hp.uniform("dropout", 0, 1),
)

def get_hyperparameter_search_space(experiment, args):
    """
    simple state function to hold the parameter search space definitions for experiments

    :param experiment: experiment name
    :return: ray config dictionary
    """
    if experiment == "rnn_tum":
        space =  dict(**BavarianCrops_parameters.__dict__,
                      **rnn_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)

    elif experiment == "transformer_tum":
        space =  dict(**BavarianCrops_parameters.__dict__,
                      **transformer_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    elif experiment == "tempcnn_tum":
        space =  dict(**BavarianCrops_parameters.__dict__,
                      **tempCNN_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    elif experiment == "msresnet_tum":
        space =  dict(**BavarianCrops_parameters.__dict__,
                      **msresnet_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    if experiment == "rnn_gaf":
        space =  dict(**GAF_parameters.__dict__,
                      **rnn_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    elif experiment == "transformer_gaf":
        space =  dict(**GAF_parameters.__dict__,
                      **transformer_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    elif experiment == "tempcnn_gaf":
        space =  dict(**GAF_parameters.__dict__,
                      **tempCNN_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    elif experiment == "msresnet_gaf":
        space =  dict(**GAF_parameters.__dict__,
                      **msresnet_parameters.__dict__)
        return space, get_points_to_evaluate(os.path.join(args.local_dir, args.experiment), args)
    else:
        raise ValueError("did not recognize experiment "+args.experiment)

def get_points_to_evaluate(path, args):
    try:
        analysis = tune.Analysis(path)
        top_runs = analysis.dataframe().sort_values(by="kappa", ascending=False).iloc[:3]
        top_runs.columns = [col.replace("config:", "") for col in top_runs.columns]

        params = top_runs[["num_layers", "dropout", "weight_decay", "learning_rate"]]

        return list(params.T.to_dict().values())
    except Exception:
        print("could not extraction previous runs from " + os.path.join(args.local_dir, args.experiment))
        return None

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

class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):

        # one iteration is five training epochs, one test epoch
        self.epochs = RAY_TEST_EVERY

        print(config)

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
        #trainer = Trainer(self.model, self.traindataloader, self.validdataloader, **config)

        if args.experiment=="transformer":
            optimizer = ScheduledOptim(
                optim.Adam(
                    filter(lambda x: x.requires_grad, self.model.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay, lr=args.learning_rate),
                self.model.d_model, args.warmup)
        else:
            optimizer = optim.Adam(
                filter(lambda x: x.requires_grad, self.model.parameters()),
                betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, lr=args.learning_rate)

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, optimizer=optimizer, **config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train five epochs and then infer once. to avoid overhead on these small datasets
        for i in range(self.epochs):
            trainstats = self.trainer.train_epoch(epoch=None)

        stats = self.trainer.test_epoch(self.validdataloader, epoch=None)
        stats.pop("inputs")
        stats.pop("ids")
        stats.pop("confusion_matrix")
        stats.pop("probas")

        stats["lossdelta"] = trainstats["loss"] - stats["loss"]
        stats["trainloss"] = trainstats["loss"]

        return stats

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

def main():
    args = parse_args()

    try:
        nruns = ray.tune.Analysis(os.path.join(args.local_dir, args.experiment)).dataframe().shape[0]
        resume=False
        todo_runs = RAY_NUM_SAMPLES - nruns
        print(f"{nruns} found in {os.path.join(args.local_dir, args.experiment)} starting remaining {todo_runs}")
        if todo_runs <= 0:
            print(f"finished all {TUNE_RUNS} runs. Increase TUNE_RUNS in databases.py if necessary. skipping tuning")
            return

    except ValueError as e:
        print(f"could not find any runs in {os.path.join(args.local_dir, args.experiment)}")
        resume=False
        todo_runs = RAY_NUM_SAMPLES

    if args.redis_address is not None:
        ray.init(redis_address=args.redis_address)
    elif not ray.is_initialized():
        ray.init(include_webui=False)

    config, points_to_evaluate = get_hyperparameter_search_space(args.experiment, args)

    args_dict = vars(args)
    config = {**config, **args_dict}
    args = Namespace(**config)

    algo = HyperOptSearch(
        config,
        max_concurrent=args.max_concurrent,
        metric="kappa",
        mode="max",
        points_to_evaluate=points_to_evaluate,
        n_initial_points=args.max_concurrent
    )


    scheduler = AsyncHyperBandScheduler(metric="kappa", mode="max",max_t=RAY_EPOCHS//RAY_TEST_EVERY,
        grace_period=HYPERBAND_GRACE_PERIOD,
        reduction_factor=HYPERBAND_REDUCTION_FACTOR,
        brackets=HYPERBAND_BRACKETS)


    analysis = tune.run(
        RayTrainer,
        config=config,
        name=args.experiment,
        num_samples=todo_runs,
        local_dir=args.local_dir,
        search_alg=algo,
        scheduler=scheduler,
        verbose=True,
        reuse_actors=False,
        resume=resume,
        checkpoint_at_end=False,
        global_checkpoint_period=9999,
        checkpoint_score_attr="kappa",
        keep_checkpoints_num=0,
        resources_per_trial=dict(cpu=args.cpu, gpu=args.gpu))

    """
        {
            args.experiment: {
                "resources_per_trial": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                'stop': {
                    'training_iteration': 1,
                    'time_total_s':3600,
                },
                "run": RayTrainer,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config,
                "local_dir":args.local_dir
            }
        },
        search_alg=algo,
        #scheduler=scheduler,
        verbose=True,)
    """

    #print("Best config is", analysis.get_best_config(metric="kappa"))
    #analysis.dataframe().to_csv("/tmp/result.csv")

if __name__=="__main__":
    main()

