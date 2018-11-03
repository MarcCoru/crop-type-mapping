import os
import json
import pandas as pd
import argparse

def load_run(path):

    result_file = os.path.join(path, "result.json")
    with open(result_file,'r') as f:
        content = f.readlines()[-1]

    result = json.loads(content)

    return result["accuracy"], result["mean_loss"], result["config"]

def load_experiment(path):
    runs = os.listdir(path)

    result = list()
    for run in runs:
        runpath = os.path.join(path, run)
        accuracy, loss, config = load_run(runpath)

        result.append(
            dict(
                accuracy=accuracy,
                loss=loss,
                batchsize=config["batchsize"],
                dataset=config["dataset"],
                hidden_dims=config["hidden_dims"],
                num_rnn_layers=config["num_rnn_layers"],
                learning_rate=config["learning_rate"]
            )
        )

    return result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment', type=str, help='Experiment. Defined as subfolder in ray root directory')
    parser.add_argument(
        '-r','--root', type=str, default=os.path.join(os.environ["HOME"], "ray_results"),
        help='ray root directory. Defaults to $HOME/ray_results')
    parser.add_argument(
        '-k','--top-k', type=int, default=1,
        help='print top K entries (default 1)')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":

    args = parse_args()

    result = load_experiment(os.path.join(args.root, args.experiment))
    result = pd.DataFrame(result).sort_values("accuracy", ascending=False)

    top = result.iloc[:args.top_k]

    print(top)


    pass