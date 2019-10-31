from argparse import Namespace
from config import HYPERPARAMETER_PATH
import pandas as pd
import os

def select_hyperparameter(experiment, hparamset):
    assert hparamset is None or isinstance(hparamset,int)

    # first hyperparameter tuning <- parameters used for some experiments, kept for backwards compatibility
    if hparamset is None:
        model = experiment.split("_")[2]
        return old_hyperparameter_config(model)
    elif isinstance(hparamset, int):
        try:
            if len(experiment.split("_")) == 4:
                name, dataset, model, meta = experiment.split("_")
            if len(experiment.split("_")) == 3:
                name, dataset, model = experiment.split("_")
            assert isinstance(name, str)
            assert isinstance(dataset, str)
            assert isinstance(model, str)
            assert dataset in ["tum", "gaf"]
        except:
            raise ValueError(f"could not parse experiment {experiment} (must be format <name>_<dataset>_<model>_<meta>) "
                             f"using hyperparameterset {hparamset} (row index of hyperparameter csv file).")

        hyperparametercsv = os.path.join(HYPERPARAMETER_PATH,f"{model}_{dataset}.csv")
        if not os.path.exists(hyperparametercsv):
            raise ValueError(f"{hyperparametercsv} does not exist")
        hparams_df = pd.read_csv(hyperparametercsv, index_col=0)
        N_rows = hparams_df.shape[0]
        if hparamset > N_rows:
            raise ValueError(f"requested hparamsset {hparamset} from {hyperparametercsv} with {N_rows} entires. "
                             f"ensure hparmaset < {N_rows}")
        hparams = hparams_df.iloc[hparamset]

        fields, dtypes = get_model_fields(model)
        configfields = [f"config/{f}" for f in fields]
        params = hparams[configfields].values
        # parse parameters in correct dtypes
        params = [dtype(p) for p,dtype in zip(params,dtypes)]
        namespace = Namespace(**dict(zip(fields,params)))
        namespace.model = model
        print(f"loaded hyperparameters {namespace} from {hyperparametercsv} (row {hparamset})")
        return namespace

def get_model_fields(model):
    assert model in ["rnn","transformer","msresnet","tempcnn"]
    if model == "rnn":
        fields = ["num_layers",
            "hidden_dims",
            "dropout",
            "weight_decay",
            "learning_rate"]
        dtypes = [int, int, float, float, float]
        return fields, dtypes
    if model == "transformer":
        fields = ["hidden_dims",
            "n_heads",
            "n_layers",
            "weight_decay",
            "learning_rate",
            "warmup",
            "dropout"]
        dtypes = [int, int, int, float, float, int, float]
        return fields, dtypes
    if model == "msresnet":
        fields = ["hidden_dims",
            "weight_decay",
            "learning_rate"]
        dtypes = [int, float, float]
        return fields, dtypes
    if model == "tempcnn":
        fields = ["kernel_size",
            "hidden_dims",
            "dropout",
            "weight_decay",
            "learning_rate"]
        dtypes = [int,int, float, float, float]
        return fields, dtypes

def old_hyperparameter_config(model):
    assert model in ["tempcnn", "transformer", "rnn", "msresnet"]
    if model == "tempcnn":
        return Namespace(
            model="tempcnn",
            kernel_size=5,
            hidden_dims=64,
            dropout=0.5,
            weight_decay=1e-6,
            learning_rate=0.001)
    if model == "transformer":
        return Namespace(
            model="transformer",
            hidden_dims = 128,
            n_heads = 3,
            n_layers = 3,
            learning_rate = 0.255410,
            dropout = 0.262039,
            weight_decay = 0.000413,
            warmup = 1000)
    if model == "rnn":
        return Namespace(
            model="rnn",
            num_layers=4,
            hidden_dims=32,
            learning_rate=0.010489,
            dropout=0.710883,
            weight_decay=0.000371,
            bidirectional=True
        )
    if model == "msresnet":
        return Namespace(
            model="msresnet",
            hidden_dims=32,
            weight_decay=0.000059,
            learning_rate=0.000657
        )
