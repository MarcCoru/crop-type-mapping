import sys
sys.path.append("..")

import unittest
import os
from utils.rayresultsparser import RayResultsParser
import pandas

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestRayResultsParser(unittest.TestCase):

    def test_load_run_rnn(self):
        parser = RayResultsParser()
        runpath = "data/tune_results/rnn/Mallat/RayTrainer_9_fold=9,hidden_dims=128,learning_rate=0.01,num_rnn_layers=1_2018-12-11_20-55-57ohafkid2"
        run = parser._load_run(runpath)
        self.assertIsInstance(run, dict)

        self.assertIsInstance(run["accuracy"], float)
        self.assertIsInstance(run["loss"], float)
        self.assertIsInstance(run["training_iteration"], int)
        self.assertIsInstance(run["timestamp"], int)
        self.assertIsInstance(run["config"], dict)

    def test_load_run_conv1d(self):
        parser = RayResultsParser()
        runpath = "data/tune_results/conv1d/CricketY/RayTrainerConv1D_0_fold=0,hidden_dims=25,learning_rate=0.1,num_layers=7,shapelet_width_increment=10_2019-01-04_15-47-51v8e6nzyv"
        run = parser._load_run(runpath)
        self.assertIsInstance(run, dict)

        self.assertIsInstance(run["accuracy"], float)
        self.assertIsInstance(run["loss"], float)
        self.assertIsInstance(run["training_iteration"], int)
        self.assertIsInstance(run["timestamp"], int)
        self.assertIsInstance(run["config"], dict)


    def test_load_all_runs_rnn(self):
        parser = RayResultsParser()
        path = 'data/tune_results/rnn/Mallat'
        resultlist = parser._load_all_runs(path=path)

        self.assertIsInstance(resultlist,list)
        self.assertIsInstance(resultlist[0],dict)
        self.assertIsInstance(resultlist[0]["accuracy"],float)
        self.assertIsInstance(resultlist[-1]["fold"], int)


    def test_load_all_runs_conv1d(self):
        parser = RayResultsParser()
        path = 'data/tune_results/conv1d/CricketY'
        resultlist = parser._load_all_runs(path=path)

        self.assertIsInstance(resultlist,list)
        self.assertIsInstance(resultlist[0],dict)
        self.assertIsInstance(resultlist[0]["accuracy"],float)
        self.assertIsInstance(resultlist[-1]["fold"], int)

    def test_get_n_best_runs_rnn(self):

        parser = RayResultsParser()
        path = 'data/tune_results/rnn/Mallat'
        best_runs_dataframe = parser._get_n_best_runs(
                         experimentpath=path,
                         n=5,
                         group_by=["hidden_dims", "learning_rate", "num_rnn_layers"])

        self.assertTrue(best_runs_dataframe["nfolds"][0] == 5)
        self.assertIsInstance(best_runs_dataframe, pandas.core.frame.DataFrame)


    def test_get_n_best_runs_conv1d(self):

        parser = RayResultsParser()
        path = 'data/tune_results/conv1d/CricketY'
        best_runs_dataframe = parser._get_n_best_runs(
                         experimentpath=path,
                         n=5,
                         group_by=["hidden_dims", "learning_rate", "num_layers", "shapelet_width_increment"])

        self.assertTrue(best_runs_dataframe["nfolds"][0] == 3)
        self.assertIsInstance(best_runs_dataframe, pandas.core.frame.DataFrame)

    def test_get_best_hyperparameters_rnn(self):
        parser = RayResultsParser()
        summary = parser.get_best_hyperparameters("data/tune_results/rnn",
                                                  outpath="/tmp/rnn",
                                                  group_by=["hidden_dims", "learning_rate", "num_rnn_layers"])

        self.assertIsInstance(summary, pandas.core.frame.DataFrame)
        self.assertTrue(os.path.exists("/tmp/rnn/hyperparams_conv1d.csv"),
                        "could not find /tmp/rnn/hyperparams_conv1d.csv")

    def test_get_best_hyperparameters_conv1d(self):
        parser = RayResultsParser()
        summary = parser.get_best_hyperparameters("data/tune_results/conv1d",
                                                  outpath="/tmp/conv1d",
                                                  group_by=["hidden_dims", "learning_rate", "num_layers", "shapelet_width_increment"])

        self.assertIsInstance(summary, pandas.core.frame.DataFrame)
        self.assertTrue(os.path.exists("/tmp/conv1d/hyperparams_conv1d.csv"),
                        "could not find /tmp/conv1d/hyperparams_conv1d.csv")



if __name__ == '__main__':
    unittest.main()