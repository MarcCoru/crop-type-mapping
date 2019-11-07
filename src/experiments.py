from argparse import Namespace

from hyperparameter import select_hyperparameter

TUM_dataset = Namespace(
    dataset = "BavarianCrops",
    trainregions = ["holl","nowa","krum"],
    testregions = ["holl","nowa","krum"],
    scheme="blocks",
    test_on = "test",
    train_on = "trainvalid",
    samplet = 70
)

TUM_dataset_random_split = Namespace(
    dataset = "BavarianCrops",
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
    features = "optical",
    scheme="random",
    test_on="test",
    train_on="train",
    samplet = 23
)



def experiments(args):

    def get_hyperparameter_args():
        return select_hyperparameter(args.experiment, args.hparamset, args.hyperparameterfolder)
    
    args.mode=None

    if args.experiment == "isprs_gaf_transformer":
        return merge([args, GAF_dataset, get_hyperparameter_args()]) #hyperparameters_transformer
    elif args.experiment == "isprs_tum_transformer":
        return merge([args, TUM_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_gaf_msresnet":
        return merge([args, GAF_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_tum_msresnet":
        return merge([args, TUM_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_gaf_rnn":
        return merge([args, GAF_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_tum_rnn":
        return merge([args, TUM_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_gaf_tempcnn":
        return merge([args, GAF_dataset, get_hyperparameter_args()])
    elif args.experiment == "isprs_tum_tempcnn":
        return merge([args, TUM_dataset, get_hyperparameter_args()])

    elif args.experiment == "isprs_rf_tum_23classes":
        args = merge([args, TUM_dataset])
        args.classmapping = "/data/BavarianCrops/classmapping.isprs.csv"
        return args
    elif args.experiment == "isprs_rf_gaf_23classes":
        args = merge([args, GAF_dataset])
        args.classmapping = "/data/BavarianCrops/classmapping.isprs.csv"
        return args
    elif args.experiment == "isprs_rf_tum_12classes":
        args = merge([args, TUM_dataset])
        args.classmapping = "/data/BavarianCrops/classmapping.isprs2.csv"
        return args
    elif args.experiment == "isprs_rf_gaf_12classes":
        args = merge([args, GAF_dataset])
        args.classmapping = "/data/BavarianCrops/classmapping.isprs2.csv"
        return args


    elif args.experiment in ["isprs_gaf_transformer_holl","isprs_gaf_tempcnn_holl","isprs_gaf_rnn_holl","isprs_gaf_msresnet_holl"]:
        args = merge([args, GAF_dataset, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args

    elif args.experiment in ["isprs_tum_transformer_all","isprs_tum_tempcnn_all","isprs_tum_rnn_all","isprs_tum_msresnet_all"]:
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["holl","nowa","krum"]
        args.testregions = ["holl"]
        return args

    elif args.experiment in ["isprs_tum_transformer_holl","isprs_tum_tempcnn_holl","isprs_tum_rnn_holl","isprs_tum_msresnet_holl"]:
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    
    elif args.experiment in ["isprs_gaf_transformer_krum","isprs_gaf_tempcnn_krum","isprs_gaf_rnn_krum","isprs_gaf_msresnet_krum"]:
        args = merge([args, GAF_dataset, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args

    elif args.experiment in ["isprs_tum_transformer_allkrum","isprs_tum_tempcnn_allkrum","isprs_tum_rnn_allkrum","isprs_tum_msresnet_allkrum"]:
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["krum","nowa","krum"]
        args.testregions = ["krum"]
        return args

    elif args.experiment in ["isprs_tum_transformer_krum","isprs_tum_tempcnn_krum","isprs_tum_rnn_krum","isprs_tum_msresnet_krum"]:
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args


    ### Model trained on different regions with block splot

    elif args.experiment == "isprs_tumholl_transformer":
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_tumkrum_transformer":
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_tumnowa_transformer":
        args = merge([args, TUM_dataset, get_hyperparameter_args()])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    elif args.experiment == "isprs_gafholl_transformer":
        args = merge([args, GAF_dataset, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_gafkrum_transformer":
        args = merge([args, GAF_dataset, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_gafnowa_transformer":
        args = merge([args, GAF_dataset, get_hyperparameter_args()])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    ### Model trained on different regions with random split
    elif args.experiment == "isprs_tumholl_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_tumkrum_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_tumnowa_transformer_randomsplit":
        args = merge([args, TUM_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args

    elif args.experiment == "isprs_gafholl_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["holl"]
        args.testregions = ["holl"]
        return args
    elif args.experiment == "isprs_gafkrum_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["krum"]
        args.testregions = ["krum"]
        return args
    elif args.experiment == "isprs_gafnowa_transformer_randomsplit":
        args = merge([args, GAF_dataset_random_split, get_hyperparameter_args()])
        args.trainregions = ["nowa"]
        args.testregions = ["nowa"]
        return args
    else:
        raise ValueError(f"Wrong experiment name {args.experiment}!")

def merge(namespaces):
    merged = dict()

    for n in namespaces:
        d = n.__dict__
        for k,v in d.items():
            merged[k]=v

    return Namespace(**merged)
