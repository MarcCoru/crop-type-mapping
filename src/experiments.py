from argparse import Namespace

from config import CLASSMAPPING

from hyperparameter import select_hyperparameter

TUM_dataset = Namespace(
    dataset = "BavarianCrops",
    classmapping = CLASSMAPPING,
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    scheme="blocks",
    test_on = "test",
    train_on = "trainvalid",
    samplet = 70
)

TUM_dataset_random_split = Namespace(
    dataset = "BavarianCrops",
    classmapping = CLASSMAPPING,
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    scheme="random",
    mode="traintest",
    test_on = "test",
    train_on = "train",
    samplet = 70
)


VNRice_dataset = Namespace(
    dataset = "VNRice",
    root="/data/vn_rice",
    mode="traintest",
    test_on = "test",
    train_on = "train",
    samplet = 50
)

GAF_dataset = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    classmapping = CLASSMAPPING,
    features = "optical",
    scheme="blocks",
    test_on="test",
    train_on="train",
    samplet = 23
)

GAF_dataset_random_split = Namespace(
    dataset = "GAFv2",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    classmapping = CLASSMAPPING,
    features = "optical",
    scheme="random",
    test_on="test",
    train_on="train",
    samplet = 23
)

def experiments(args):
    
    args.mode=None

    if args.experiment == "isprs_gaf_transformer":
        return merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)]) #hyperparameters_transformer
    elif args.experiment == "isprs_tum_transformer":
        return merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_gaf_msresnet":
        return merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_tum_msresnet":
        return merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_gaf_rnn":
        return merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_tum_rnn":
        return merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_gaf_tempcnn":
        return merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
    elif args.experiment == "isprs_tum_tempcnn":
        return merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])

    elif args.experiment == "isprs_svm_tum":
        return merge([args, TUM_dataset])
    elif args.experiment == "isprs_svm_gaf":
        return merge([args, GAF_dataset])
    elif args.experiment == "isprs_rf_tum":
        return merge([args, TUM_dataset])
    elif args.experiment == "isprs_rf_gaf":
        return merge([args, GAF_dataset])


    ### Model trained on different regions with block splot

    elif args.experiment == "isprs_tumholl_transformer":
        args = merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_tumkrum_transformer":
        args = merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_tumnowa_transformer":
        args = merge([args, TUM_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    elif args.experiment == "isprs_gafholl_transformer":
        args = merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_gafkrum_transformer":
        args = merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_gafnowa_transformer":
        args = merge([args, GAF_dataset, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    ### Model trained on different regions with random split
    elif args.experiment == "isprs_tumholl_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_tumkrum_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_tumnowa_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    elif args.experiment == "isprs_gafholl_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_gafkrum_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_gafnowa_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, select_hyperparameter(args.experiment, args.hparamset)])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args
    else:
        raise ValueError(f"Wrong experiment name {args.experiment}!")


    """Experiment Modalities
    # checkout 83d998dc30abc83f5ca0316aea5baff5133846ba
    if args.experiment == "tumgaf_gaf_transformer_optical":
        args = merge([args, GAF_dataset, hyperparameters_transformer])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_transformer_radar":
        args = merge([args, GAF_dataset, hyperparameters_transformer])
        args.features="radar"

    elif args.experiment == "tumgaf_gaf_transformer_all":
        args = merge([args, GAF_dataset, hyperparameters_transformer])
        args.features="all"

    elif args.experiment == "tumgaf_gaf_tempcnn_all":
        args = merge([args, GAF_dataset])
        args.model = "tempcnn"

    elif args.experiment == "vnrice_rnn":
        args = merge([args, VNRice_dataset, hyperparameters_rnn])

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

    elif args.experiment == "tumgaf_gafall_transformer":
        args = merge([args, GAF_dataset_allregions, hyperparameters_transformer])

    elif args.experiment == "tumgaf_gafall_msresnet":
        args = merge([args, GAF_dataset_allregions, hyperparameters_msresnet])

    elif args.experiment == "tumgaf_gafall_rnn":
        args = merge([args, GAF_dataset_allregions, hyperparameters_rnn])

    elif args.experiment == "tumgaf_gafall_tempcnn":
        args = merge([args, GAF_dataset_allregions, hyperparameters_tempcnn])

    elif args.experiment == "tumgaf_gaf_transformer":
        args = merge([args, GAF_dataset, hyperparameters_transformer])

    elif args.experiment == "tumgaf_gaf_rnn_optical":
        args = merge([args, GAF_dataset, hyperparameters_rnn])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_tempcnn_optical":
        args = merge([args, GAF_dataset, hyperparameters_tempCNN])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_transformer_optical":
        args = merge([args, GAF_dataset, hyperparameters_transformer])
        args.features="optical"

    elif args.experiment == "tumgaf_gaf_msresnet_optical":
        args = merge([args, GAF_dataset, hyperparameters_msresnet])
        args.features="optical"
    """


def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)
