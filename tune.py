import ray
import ray.tune as tune
import train
import os, sys, io
import contextlib
from models.DualOutputRNN import DualOutputRNN
from utils.UCR_Dataset import UCRDataset
from utils.classmetric import ClassMetric
import torch
import numpy as np
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
import argparse
from hyperopt import hp

class TrainDualOutputRNN(ray.tune.Trainable):
    def _setup(self, config):
        batchsize = config["batchsize"]
        workers = config["workers"]
        hidden_dims = config["hidden_dims"]
        learning_rate = config["learning_rate"]
        num_rnn_layers = config["num_rnn_layers"]
        dataset = config["dataset"]

        traindataset = UCRDataset(dataset, partition="trainvalid", ratio=.75, randomstate=2, silent=True)
        validdataset = UCRDataset(dataset, partition="test", ratio=.75, randomstate=2, silent=True)
        nclasses = traindataset.nclasses

        # handles multitxhreaded batching and shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                      num_workers=workers, pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=batchsize, shuffle=False,
                                                      num_workers=workers, pin_memory=False)

        self.model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=hidden_dims, num_rnn_layers=num_rnn_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train_epoch(self):

        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.traindataloader.dataset.nclasses)

        logged_loss_early = list()
        logged_loss_class = list()
        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, targets = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            loss, logprobabilities = self.model.loss(inputs, targets)
            logged_loss_class.append(loss.detach().cpu().numpy())

            maxclass = logprobabilities.argmax(1)
            prediction = maxclass.mode(1)[0]

            stats = metric(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

            loss.backward()
            self.optimizer.step()

        stats["loss_early"] = np.array(logged_loss_early).mean()
        stats["loss_class"] = np.array(logged_loss_class).mean()

        return stats

    def test_epoch(self):
        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.validdataloader.dataset.nclasses)

        logged_loss_early = list()
        logged_loss_class = list()
        with torch.no_grad():
            for iteration, data in enumerate(self.validdataloader):

                inputs, targets = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                loss, logprobabilities = self.model.loss(inputs, targets)
                logged_loss_class.append(loss.detach().cpu().numpy())

                maxclass = logprobabilities.argmax(1)
                prediction = maxclass.mode(1)[0]

                stats = metric(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

        stats["mean_loss"] = np.array(logged_loss_class).mean()

        return stats

    def _train(self):
        self.train_epoch()
        return self.test_epoch()

    def _save(self, path):
        path = path + "_model.pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        self.model.load_state_dict(path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--dataset', type=str, default="Trace", help='UCR Dataset. Will also name the experiment')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-c', '--cpu', type=int, default=2, help='number of CPUs allocated per trial run (default 2)')
    parser.add_argument(
        '-g', '--gpu', type=float, default=.25,
        help='number of GPUs allocated per trial run (can be float for multiple runs sharing one GPU, default 0.25)')
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":
    args = parse_args()

    ray.init(include_webui=False)

    config = dict(
        batchsize=32,
        workers=0,
        epochs=999,
        hidden_dims=tune.grid_search([2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10]),
        learning_rate=tune.grid_search([1e-2, 1e-3]),
        earliness_factor=1,
        switch_epoch=999,
        num_rnn_layers=tune.grid_search([1, 2, 3, 4]),
        dataset=args.dataset,
        savepath="/home/marc/tmp/model_r1024_e4k.pth",
        loadpath=None,
        silent=True)

    config = dict(
        batchsize=args.batchsize,
        workers=2,
        hidden_dims=hp.choice("hidden_dims", [2**4,2**5,2**6,2**7,2**8,2**9]),
        learning_rate=hp.uniform("learning_rate", 1e-3,1e-1),
        num_rnn_layers=hp.choice("num_rnn_layers", [1,2,3,4]),
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
                "run": TrainDualOutputRNN,
                "num_samples": 1 if args.smoke_test else 300,
                "checkpoint_at_end": True,
                "config": config
            }
        },
        verbose=0,
        search_alg=algo,
        scheduler=hb_scheduler)