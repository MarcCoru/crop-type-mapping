import os
from argparse import Namespace

hyperparameters_rnn = Namespace(
    num_layers=4,
    hidden_dims=32,
    learning_rate=0.010489,
    dropout=0.710883,
    weight_decay=0.000371
)


def experiments(args):

    args.samplet = 50
    args.trainids = None
    args.testids = None

    """Experiment Modalities"""
    if args.experiment == "tumgaf_gaf_rnn_optical":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.features="optical"
        args.num_layers = hyperparameters_rnn.num_layers
        args.hidden_dims = hyperparameters_rnn.hidden_dims
        args.learning_rate = hyperparameters_rnn.learning_rate
        args.weight_decay = hyperparameters_rnn.weight_decay
        args.dropout = hyperparameters_rnn.dropout
        args.bidirectional = True

    elif args.experiment == "tumgaf_gaf_rnn_radar":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = hyperparameters_rnn.num_layers
        args.hidden_dims = hyperparameters_rnn.hidden_dims
        args.learning_rate = hyperparameters_rnn.learning_rate
        args.weight_decay = hyperparameters_rnn.weight_decay
        args.dropout = hyperparameters_rnn.dropout
        args.bidirectional = True
        args.features="radar"

    elif args.experiment == "tumgaf_gaf_rnn_all":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = hyperparameters_rnn.num_layers
        args.hidden_dims = hyperparameters_rnn.hidden_dims
        args.learning_rate = hyperparameters_rnn.learning_rate
        args.weight_decay = hyperparameters_rnn.weight_decay
        args.dropout = hyperparameters_rnn.dropout
        args.bidirectional = True
        args.features="all"

    elif args.experiment == "tumgaf_gaf_tempcnn_all":
        args.model = "tempcnn"
        args.dataset = "GAFv2"
        args.features = "all"

        """Models and Datasets"""
    elif args.experiment == "tumgaf_tum_rnn":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "test"
        args.train_on = "train"
        args.num_layers = hyperparameters_rnn.num_layers
        args.hidden_dims = hyperparameters_rnn.hidden_dims
        args.learning_rate = hyperparameters_rnn.learning_rate
        args.weight_decay = hyperparameters_rnn.weight_decay
        args.dropout = hyperparameters_rnn.dropout
        args.bidirectional = True

    elif args.experiment == "tumgaf_gaf_rnn":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = hyperparameters_rnn.num_layers
        args.hidden_dims = hyperparameters_rnn.hidden_dims
        args.learning_rate = hyperparameters_rnn.learning_rate
        args.weight_decay = hyperparameters_rnn.weight_decay
        args.dropout = hyperparameters_rnn.dropout
        args.bidirectional = True
        args.features = "all"

    elif args.experiment == "tumgaf_tum_msresnet":
        args.model = "msresnet"
        args.dataset = "GAFv2"

        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "test"
        args.train_on = "train"

    elif args.experiment == "tumgaf_tum_tempcnn":
        args.model = "tempcnn"
        args.dataset = "GAFv2"

        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "test"
        args.train_on = "train"

    elif args.experiment == "tumgaf_gaf_msresnet":
        args.model = "msresnet"
        args.dataset = "GAFv2"
        args.features = "all"

    elif args.experiment == "tumgaf_gaf_tempcnn":
        args.model = "tempcnn"
        args.dataset = "GAFv2"
        args.features = "all"

    elif args.experiment == "tumgaf_tum_transformer":

        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "test"
        args.train_on = "train"

        args.model = "transformer"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4

    elif args.experiment == "tumgaf_gaf_transformer":
        args.dataset = "GAFv2"
        args.features = "all"

        args.model = "transformer"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4

        """OLD experiments"""
    elif args.experiment == "test":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.num_layers = 3
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/random/holl_2018_mt_pilot_train.txt"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/random/holl_2018_mt_pilot_test.txt"
        args.hidden_dims = 128
        args.bidirectional = True
        args.test_on = "test"
        args.train_on = "train"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]

    elif args.experiment == "TUM_ALL_rnn_allclasses":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping83.csv"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["HOLL_2018_MT_pilot","KRUM_2018_MT_pilot","NOWA_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]

    elif args.experiment == "TUM_ALL_transformer_allclasses":
        args.model = "transformer"
        args.dataset = "BavarianCrops"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4
        args.trainregions = ["HOLL_2018_MT_pilot","KRUM_2018_MT_pilot","NOWA_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping83.csv"

    elif args.experiment == "TUM_ALL_rnn":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["HOLL_2018_MT_pilot","KRUM_2018_MT_pilot","NOWA_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]

    elif args.experiment == "TUM_GEN_rnn":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["HOLL_2018_MT_pilot","KRUM_2018_MT_pilot"]
        args.testregions = ["NOWA_2018_MT_pilot"]

    elif args.experiment == "BreizhCrops_rnn":
        args.model = "rnn"
        args.dataset = "BreizhCrops"
        args.classmapping = None
        args.num_layers = 3
        args.samplet = 45
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["frh01", "frh02", "frh03"]
        args.testregions = ["frh04"]

    elif args.experiment == "BreizhCrops_transformer":
        args.model = "transformer"
        args.dataset = "BreizhCrops"
        args.hidden_dims = 128
        args.samplet = 45
        args.n_heads = 4
        args.n_layers = 4
        args.trainregions = ["frh01", "frh02", "frh03"]
        args.testregions = ["frh04"]

    elif args.experiment == "TUM_HOLL_rnn":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]

    elif args.experiment == "TUM_HOLL_transformer":
        args.model = "transformer"
        args.dataset = "BavarianCrops"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"

    elif args.experiment == "TUM_ALL_transformer":
        args.model = "transformer"
        args.dataset = "BavarianCrops"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4
        args.trainregions = ["HOLL_2018_MT_pilot","KRUM_2018_MT_pilot","NOWA_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"

    elif args.experiment == "GAFHDF5_transformer":
        args.model = "transformer"
        args.dataset = "GAFHDF5"
        args.hidden_dims = 256
        args.n_heads = 8
        args.n_layers = 6

    else:
        raise ValueError("Wrong experiment name!")

    return args