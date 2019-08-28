import os
from argparse import Namespace

hyperparameters_rnn = Namespace(
    model="rnn",
    num_layers=4,
    hidden_dims=32,
    learning_rate=0.010489,
    dropout=0.710883,
    weight_decay=0.000371,
    bidirectional = True
)

hyperparameters_transformer = Namespace(
    model="transformer",
    hidden_dims = 128,
    n_heads = 3,
    n_layers = 3,
    learning_rate = 0.255410,
    dropout = 0.262039,
    weight_decay = 0.000413,
    warmup = 1000
)

hyperparameters_msresnet = Namespace(
    model="msresnet",
    hidden_dims=32,
    weight_decay=0.000059,
    learning_rate=0.000657
)

hyperparameters_tempCNN = Namespace(
    model="tempcnn",
    kernel_size=5,
    hidden_dims=64,
    dropout=0.5,
    weight_decay=1e-6,
    learning_rate=0.001
)

TUM_dataset = Namespace(
    dataset = "BavarianCrops",
    classmapping = "/data/BavarianCrops/classmapping.csv.gaf.v2",
    trainregions = ["holl"],
    testregions = ["holl"],
    mode="traintest",
    test_on = "test",
    train_on = "train",
    samplet = 70
)

GAF_dataset = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl"],
    testregions = ["holl"],
    classmapping = "/data/BavarianCrops/classmapping.csv.gaf.v2",
    features = "all",
    test_on="test",
    train_on="train",
    samplet = 23
)

def experiments(args):

    merge([args, TUM_dataset, hyperparameters_transformer])

    """Experiment Modalities"""
    if args.experiment == "tumgaf_gaf_transformer_optical":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_transformer_radar":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="radar"

    elif args.experiment == "tumgaf_gaf_transformer_all":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="all"

    elif args.experiment == "tumgaf_gaf_tempcnn_all":
        args = merge([args, GAF_dataset])
        args.model = "tempcnn"

        """Models and Datasets"""
    elif args.experiment == "tumgaf_tum_rnn":
        args = merge([args, TUM_dataset, hyperparameters_rnn])

    elif args.experiment == "tumgaf_gaf_rnn":
        args = merge([args, GAF_dataset, hyperparameters_rnn])

    elif args.experiment == "tumgaf_tum_msresnet":
        args = merge([args, TUM_dataset, hyperparameters_msresnet])

    elif args.experiment == "tumgaf_tum_tempcnn":
        args = merge([args, TUM_dataset, hyperparameters_tempCNN])

    elif args.experiment == "tumgaf_gaf_msresnet":
        args = merge([args, GAF_dataset, hyperparameters_msresnet])

    elif args.experiment == "tumgaf_gaf_tempcnn":
        args = merge([args, GAF_dataset, hyperparameters_tempCNN])

    elif args.experiment == "tumgaf_tum_transformer":
        args = merge([args, TUM_dataset, hyperparameters_transformer])

    elif args.experiment == "tumgaf_gaf_transformer":
        args = merge([args, GAF_dataset, hyperparameters_transformer])

    elif args.experiment == "tumgaf_gaf_rnn_optical":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_tempcnn_optical":
        args = merge([args, GAF_dataset, hyperparameters_tempcnn])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_transformer_optical":
        args = merge([args, GAF_dataset, hyperparameters_transformer])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_msresnet_optical":
        args = merge([args, GAF_dataset, hyperparameters_msresnet])
        args.features="optical"

    else:
        raise ValueError("Wrong experiment name!")

    return args


def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)
