import sys
sys.path.append("..")

import ray
import ray.tune as tune
from hyperopt import hp
from ray.tune.schedulers import HyperBandScheduler
from tune import RayTrainer
from tune_mori_datasets_conv1d import RayTrainerConv1D
import unittest

class TestTune(unittest.TestCase):

    def test_Tune_Trace(self):

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            dataset = "Trace"
            # tune.grid_search(
            config = dict(
                batchsize=2,
                workers=2,
                epochs=20,
                switch_epoch=9999,
                fold=1,
                hidden_dims=hp.choice("hidden_dims", [2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]),
                learning_rate=hp.uniform("learning_rate", 1e-3, 1e-1),
                data_noise=hp.uniform("data_noise", 0, 1e-1),
                num_rnn_layers=hp.choice("num_rnn_layers", [1, 2, 3, 4]),
                earliness_factor=.75,
                dataset="Trace")

            hb_scheduler = HyperBandScheduler(
                time_attr="training_iteration",
                reward_attr="accuracy",
                max_t=1)

            experiment_name = dataset

            algo = ray.tune.suggest.HyperOptSearch(space=config, max_concurrent=30, reward_attr="neg_mean_loss")

            tune.run_experiments(
                {
                    experiment_name: {
                        "trial_resources": {
                            "cpu": 2,
                            "gpu": 1,
                        },
                        "run": RayTrainer,
                        "num_samples": 1,
                        "checkpoint_at_end": False,
                        "config": config
                    }
                },
                verbose=0,
                search_alg=algo,
                scheduler=hb_scheduler)

        except Exception as e:
            self.fail("Failed Tune: " + str(e))

    def test_Tune_Trace_Conv1D(self):

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)
            dataset = "Trace"
            # tune.grid_search(
            config = dict(
                batchsize=32,
                workers=2,
                epochs=99999,
                switch_epoch=9999,
                earliness_factor=1,
                fold=tune.grid_search([0]),
                hidden_dims=tune.grid_search([25]),
                learning_rate=tune.grid_search([1e-2]),
                num_layers=tune.grid_search([1, 2]),
                dataset=dataset)

            experiment_name = dataset

            tune.run_experiments(
                {
                    experiment_name: {
                        "trial_resources": {
                            "cpu": 2,
                            "gpu": 1,
                        },
                        'stop': {
                            'training_iteration': 1,
                            'time_total_s': 1,
                        },
                        "run": RayTrainerConv1D,
                        "num_samples": 1,
                        "checkpoint_at_end": False,
                        "config": config
                    }
                },
                # local_dir=args.local_dir,
                # scheduler=ahb_scheduler,
                verbose=0)  #

        except Exception as e:
            self.fail("Failed Tune: " + str(e))

if __name__ == '__main__':
    unittest.main()