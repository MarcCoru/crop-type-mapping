import ray
import ray.tune as tune
from models.DualOutputRNN import DualOutputRNN
from utils.UCR_Dataset import UCRDataset
import torch
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
import argparse
from hyperopt import hp
from utils.trainer import Trainer
from utils.parse_rayresults import parse_experiment

class RayTrainer(ray.tune.Trainable):
    def _setup(self, config):

        traindataset = UCRDataset(config["dataset"],
                                  partition="trainvalid",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True,
                                  augment_data_noise=0)

        validdataset = UCRDataset(config["dataset"],
                                  partition="test",
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
    args, _ = parser.parse_known_args()
    return args

def tune_dataset(args):
    """designed to to tune on the same datasets as used by Mori et al. 2017"""

    config = dict(
        batchsize=args.batchsize,
        workers=2,
        epochs=99999,
        switch_epoch=9999,
        earliness_factor=1,
        fold=tune.grid_search([0, 1, 2, 3, 4]),
        hidden_dims=tune.grid_search([2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
        learning_rate=tune.grid_search([1e-2,1e-3,1e-4]),
        dropout=0.3,
        num_rnn_layers=tune.grid_search([1,2,3,4]),
        dataset=args.dataset)

    hb_scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="accuracy",
        max_t=1 if args.smoke_test else 6)

    ahb_scheduler = AsyncHyperBandScheduler(
        reward_attr="accuracy",
        time_attr="training_iteration",
        max_t=10 if not args.smoke_test else 1,
        grace_period=1,
        reduction_factor=2,
        brackets=3
        )

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
                "run": RayTrainer,
                "num_samples": 1,
                "checkpoint_at_end": False,
                "config": config
            }
        },
        #local_dir=args.local_dir,
        #scheduler=ahb_scheduler,
        verbose=0) #
        #
        #

if __name__=="__main__":
    import datetime
    import os
    import sys

    # parse input arguments
    args = parse_args()

    datasets = [dataset.strip() for dataset in open("experiments/morietal2017/datasets.txt", 'r').readlines()]
    
    resultsdir = os.path.join(os.getenv("HOME"),"ray_results")
       
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
            top = parse_experiment(experimentpath=os.path.join(resultsdir,dataset),outcsv=os.path.join(resultsdir,dataset,"params.csv"))
            num_hidden, learning_rate, num_rnn_layers = top.iloc[0].name
            param_string = "num_hidden:{}, learning_rate:{}, num_rnn_layers:{}".format(*top.iloc[0].name)
            perf_string = "accuracy {:.2f} (+-{:.2f}) in {:.0f} folds".format(top.iloc[0].mean_accuracy,
                                                                top.iloc[0].std_accuracy, top.iloc[0].nfolds)
            print("{time} finished tuning dataset {dataset} {perf_string}, {param_perf_string}".format(
                time=time,
                dataset=dataset,
                perf_string=perf_string,
                param_string=param_string),
                file=open(os.path.join(resultsdir, "datasets.log"), "a"))

        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print("error" + str(e))
            pass
        finally:
            pass
