import sys
sys.path.append("..")

import ray
import unittest
from tune import get_hyperparameter_search_space, tune_dataset_rnn, tune_mori_datasets
import logging
import shutil
import os

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TestTune(unittest.TestCase):

    def test_Tune_Trace_RNN(self):

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            args = Namespace(
                batchsize=32,
                cpu=2,
                dataset='Trace',
                experiment='test_rnn',
                gpu=0,
                local_dir='/tmp',
                skip_processed=False,
                smoke_test=True)
            config = get_hyperparameter_search_space(args.experiment, args)
            tune_dataset_rnn(args,config)
        except Exception as e:
            self.fail(self.fail(logging.exception(e)))

    def test_Tune_Trace_Conv1d(self):

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            args = Namespace(
                batchsize=32,
                cpu=2,
                dataset='Trace',
                experiment='test_conv1d',
                gpu=0,
                local_dir='/tmp',
                skip_processed=False,
                smoke_test=True)
            config = get_hyperparameter_search_space(args.experiment, args)
            tune_dataset_rnn(args,config)
        except Exception as e:
            self.fail(self.fail(logging.exception(e)))

    def test_Tune_FiftyWords_Conv1d(self):

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            args = Namespace(
                batchsize=32,
                cpu=2,
                dataset='FiftyWords',
                experiment='test_conv1d',
                gpu=0,
                local_dir='/tmp',
                skip_processed=False,
                smoke_test=True)
            config = get_hyperparameter_search_space(args.experiment, args)
            tune_dataset_rnn(args,config)
        except Exception as e:
            self.fail(self.fail(logging.exception(e)))

    def test_tune_mori_datasets_conv1d(self):
        if os.path.exists("/tmp/test_conv1d"):
            shutil.rmtree("/tmp/test_conv1d")

        # create fake dataset.txt
        datasetfile = "/tmp/test_conv1d_datasets.txt"
        with open(datasetfile,"w") as file:
            file.write("FiftyWords\nTrace\nTwoPatterns")

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            args = Namespace(
                batchsize=32,
                cpu=2,
                datasetfile=datasetfile,
                experiment='test_conv1d',
                gpu=0,
                local_dir='/tmp',
                skip_processed=False,
                smoke_test=True)
            tune_mori_datasets(args)

            self.assertTrue(os.path.exists("/tmp/test_conv1d/datasets.log"))
            self.assertTrue(os.path.exists("/tmp/test_conv1d/Trace/params.csv"))
            self.assertTrue(os.path.exists("/tmp/test_conv1d/TwoPatterns/params.csv"))
            self.assertTrue(os.path.exists("/tmp/test_conv1d/FiftyWords/params.csv"))
        except Exception as e:
            self.fail(self.fail(logging.exception(e)))

    def test_tune_mori_datasets_rnn(self):
        if os.path.exists("/tmp/test_rnn"):
            shutil.rmtree("/tmp/test_rnn")

        # create fake dataset.txt
        datasetfile = "/tmp/test_rnn_datasets.txt"
        with open(datasetfile,"w") as file:
            file.write("Trace\nTwoPatterns")

        try:
            if not ray.is_initialized():
                ray.init(include_webui=False)

            args = Namespace(
                batchsize=32,
                cpu=2,
                datasetfile=datasetfile,
                experiment='test_rnn',
                gpu=0,
                local_dir='/tmp',
                skip_processed=False,
                smoke_test=True)
            tune_mori_datasets(args)

            self.assertTrue(os.path.exists("/tmp/test_rnn/datasets.log"))
            self.assertTrue(os.path.exists("/tmp/test_rnn/Trace/params.csv"))
            self.assertTrue(os.path.exists("/tmp/test_rnn/TwoPatterns/params.csv"))
        except Exception as e:
            self.fail(self.fail(logging.exception(e)))

if __name__ == '__main__':
    unittest.main()