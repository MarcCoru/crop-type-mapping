from eval import eval
import argparse
import pandas as pd
import os

"""
Calls eval for each dataset given hyperparameterfile provided by tune.py
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-m', '--model', type=str, default="DualOutputRNN", help='Model variant')
    parser.add_argument(
        '-e', '--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '--dropout', type=float, default=.2, help='dropout probability of the rnn layer')
    parser.add_argument(
        '-n', '--num_rnn_layers', type=int, default=1, help='number of RNN layers')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of RNN hidden dimensions')
    parser.add_argument(
        '-a','--earliness_factor', type=float, default=1, help='earliness factor')
    parser.add_argument(
        '--hparams', type=str, default=None, help='hyperparams csv file')
    parser.add_argument(
        '--root', type=str, default="/data/remote/early_rnn/", help='root folder')
    parser.add_argument(
        '--experiment', type=str, default="experiment", help='experiment subfolder in root')
    parser.add_argument(
        '--weight_folder', type=str, default=None, help='folder containing model weights to restore from. '
                                                        'needs to follow the structure <weight_folder>/<dataset>/run/model_<epochs>.pth')
    parser.add_argument(
        '--load_epoch', type=int, default=29, help='load epoch from the path files in the weight_folder...'
                                                   'a file <weight_folder>/<dataset>/run/model_<epochs>.pth'
                                                   'with the corresponding epoch must exist')
    parser.add_argument(
        '--resume', action='store_true', help='skip previously processed runs')
    parser.add_argument(
        '-i', '--show-n-samples', type=int, default=2, help='show n samples in visdom')
    parser.add_argument(
        '-s', '--switch_epoch', type=int, default=None, help='epoch at which to switch the loss function '
                                                             'from classification training to early training')
    parser.add_argument(
        '--loss_mode', type=str, default="twophase_early_linear", help='which loss function to choose. '
                                                                       'valid options are early_reward,  '
                                                                       'twophase_early_reward, '
                                                                       'twophase_linear_loss, or twophase_cross_entropy')

    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":

    args = parse_args()
    root = args.root #
    csvfile = os.path.join(root,"eval_results.csv")
    resume = args.resume

    hparams = pd.read_csv(args.hparams).set_index("dataset")

    if not resume:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(csvfile, index_col=0)

    for dataset, params in hparams.iterrows():
        print(dataset)

        if dataset in df.index:
            print("skipping "+dataset)
            continue

        # get hyperparameters from the hyperparameter file for the current dataset...
        args.hidden_dims = int(params["hidden_dims"])
        args.learning_rate = params["learning_rate"]
        args.num_rnn_layers = int(params["num_rnn_layers"])

        args.store = os.path.join(os.path.join(root,args.experiment,"models",dataset))
        visdomenv = args.experiment+"_"+dataset

        if args.weight_folder is not None:
            load_weights = os.path.join(args.weight_folder, dataset, "run", "model_{epoch}.pth".format(epoch=args.load_epoch))
        else:
            load_weights = None

        try:
            logged_data = eval(
                dataset = dataset,
                batchsize = args.batchsize,
                workers = args.workers,
                num_rnn_layers = args.num_rnn_layers,
                dropout = args.dropout,
                hidden_dims = args.hidden_dims,
                epochs = args.epochs,
                store = args.store,
                switch_epoch = args.switch_epoch,
                learning_rate = args.learning_rate,
                earliness_factor = args.earliness_factor,
                loss_mode=args.loss_mode,
                visdomenv=visdomenv,
                load_weights=load_weights
            )

            last_row = logged_data.iloc[-1]
            last_row.name = dataset
            df = df.append(last_row)
            df.to_csv(csvfile)
        except Exception as e:
            print(e)
            pass

