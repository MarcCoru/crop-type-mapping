import ray
import ray.tune as tune
from models.DualOutputRNN import DualOutputRNN
from utils.UCR_Dataset import UCRDataset
import torch
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
import argparse
from hyperopt import hp
from utils.trainer import Trainer

class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):

        traindataset = UCRDataset(config["dataset"],
                                  partition="train",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True,
                                  augment_data_noise=config["data_noise"])

        validdataset = UCRDataset(config["dataset"],
                                  partition="valid",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True)
        nclasses = traindataset.nclasses

        # handles multitxhreaded batching andconfig shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
                                                           num_workers=config["workers"],
                                                           pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
                                                      num_workers=config["workers"], pin_memory=False)

        self.model = DualOutputRNN(input_dim=1,
                                   nclasses=nclasses,
                                   hidden_dim=config["hidden_dims"],
                                   num_rnn_layers=config["num_rnn_layers"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train five epochs and then infer once. to avoid overhead on these small datasets
        for i in range(5):
            self.trainer.train_epoch(epoch=None)

        return self.trainer.test_epoch(epoch=None)

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--dataset', type=str, default="Trace", help='UCR Dataset. Will also name the experiment')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
    parser.add_argument(
        '-a', '--earliness_factor', type=float, default=.75, help='earliness factor')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.25,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

def tune_dataset(args):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    config = dict(
        batchsize=args.batchsize,
        workers=2,
        epochs=20,
        switch_epoch=9999,
        fold=1,
        hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
        learning_rate=hp.uniform("learning_rate", 1e-4, 1e-2),
        data_noise=hp.uniform("data_noise", 0, 1e-2),
        dropout=hp.uniform("dropout", 0, 0.5),
        num_rnn_layers=hp.choice("num_rnn_layers", [1, 2, 3, 4]),
        dataset=args.dataset)

    hb_scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="accuracy",
        max_t=1 if args.smoke_test else 6)

    ahb_scheduler = AsyncHyperBandScheduler(
        reward_attr="accuracy",
        time_attr="training_iteration",
        max_t=6,
        grace_period=1,
        reduction_factor=2,
        brackets=3
        )

    experiment_name = args.dataset

    """
    requires HyperOpt to be installed from source:
    git clone https://github.com/hyperopt/hyperopt.git
    python setup.py build
    python setup.py install
    """
    algo = ray.tune.suggest.HyperOptSearch(space=config, max_concurrent=30, reward_attr="accuracy")

    tune.run_experiments(
        {
            experiment_name: {
                "trial_resources": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                "run": RayTrainer,
                "num_samples": 1 if args.smoke_test else 300,
                "checkpoint_at_end": True,
                "config": config
            }
        },
        verbose=0,
        search_alg=algo,
        scheduler=ahb_scheduler)


def main(args):
    #tune.grid_search(

    config = dict(
        batchsize=args.batchsize,
        workers=2,
        epochs=20,
        switch_epoch=9999,
        fold=1,
        hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
        learning_rate=hp.uniform("learning_rate", 1e-3, 1e-1),
        data_noise=hp.uniform("data_noise", 0, 1e-1),
        num_rnn_layers=hp.choice("num_rnn_layers", [1, 2, 3, 4]),
        earliness_factor=args.earliness_factor,
        dataset=args.dataset)

    hb_scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="accuracy",
        max_t=1 if args.smoke_test else 30)

    ahb_scheduler = AsyncHyperBandScheduler(
        reward_attr="accuracy",
        time_attr="training_iteration",
        max_t=60,
        grace_period=3,
        reduction_factor=2,
        brackets=5
        )

    experiment_name = args.dataset

    """
    requires HyperOpt to be installed from source:
    git clone https://github.com/hyperopt/hyperopt.git
    python setup.py build
    python setup.py install
    """
    algo = ray.tune.suggest.HyperOptSearch(space=config, max_concurrent=30, reward_attr="neg_mean_loss")

    tune.run_experiments(
        {
            experiment_name: {
                "trial_resources": {
                    "cpu": args.cpu,
                    "gpu": args.gpu,
                },
                "run": RayTrainer,
                "num_samples": 1 if args.smoke_test else 300,
                "checkpoint_at_end": True,
                "config": config
            }
        },
        verbose=0,
        search_alg=algo,
        scheduler=hb_scheduler)


if __name__=="__main__":

    # parse input arguments
    args = parse_args()

    # start ray server
    ray.init(include_webui=False)

    # tune dataset
    tune_dataset(args)
