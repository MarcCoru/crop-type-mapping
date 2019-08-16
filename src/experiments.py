import os

def experiments(args):

    args.samplet = 50
    args.trainids = None
    args.testids = None

    """Experiment Modalities"""
    if args.experiment == "tumgaf_gaf_rnn_optical":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.features="optical"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True

    elif args.experiment == "tumgaf_gaf_rnn_radar":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.features="radar"

    elif args.experiment == "tumgaf_gaf_rnn_all":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.features="all"

        """Models and Datasets"""
    elif args.experiment == "tumgaf_tum_rnn":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "eval"
        args.train_on = "trainvalid"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True

    elif args.experiment == "tumgaf_gaf_rnn":
        args.model = "rnn"
        args.dataset = "GAFv2"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.features = "optical"

    elif args.experiment == "tumgaf_tum_msresnet":
        args.model = "msresnet"
        args.dataset = "GAFv2"

        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "eval"
        args.train_on = "trainvalid"

    elif args.experiment == "tumgaf_gaf_msresnet":
        args.model = "msresnet"
        args.dataset = "GAFv2"
        args.features = "optical"

    elif args.experiment == "tumgaf_tum_transformer":

        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf.v2"
        args.trainregions = ["HOLL_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot"]
        args.trainids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_trainids.csv"
        args.testids = os.getenv("HOME") + "/data/BavarianCrops/ids/gaf_holl_testids.csv"
        args.test_on = "eval"
        args.train_on = "trainvalid"

        args.model = "transformer"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4

    elif args.experiment == "tumgaf_gaf_transformer":
        args.dataset = "GAFv2"
        args.features = "optical"

        args.model = "transformer"
        args.hidden_dims = 256
        args.samplet = 30
        args.n_heads = 4
        args.n_layers = 4

        """OLD experiments"""
    elif args.experiment == "test":
        args.model = "rnn"
        args.dataset = "BavarianCrops"
        args.classmapping = os.getenv("HOME") + "/data/BavarianCrops/classmapping.csv.gaf"
        args.num_layers = 3
        args.hidden_dims = 128
        args.bidirectional = True
        args.trainregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]
        args.testregions = ["HOLL_2018_MT_pilot", "KRUM_2018_MT_pilot", "NOWA_2018_MT_pilot"]

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