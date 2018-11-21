import torch
from models.DualOutputRNN import DualOutputRNN
#from models.AttentionRNN import AttentionRNNs
from utils.UCR_Dataset import UCRDataset
from utils.Synthetic_Dataset import SyntheticDataset
from utils.classmetric import ClassMetric
from utils.logger import Printer, VisdomLogger, Logger
import argparse
import numpy as np


class Trainer():

    def __init__(self,model, traindataloader, validdataloader, config):

        self.epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        self.earliness_factor = config["earliness_factor"]
        self.switch_epoch = config["switch_epoch"]

        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses

        self.visdom = VisdomLogger(env=config["visdomenv"])
        self.logger = Logger(columns=["accuracy"], modes=["train", "test"])
        self.lossmode = config["loss_mode"] # early_reward,  twophase_early_reward, twophase_linear_loss, or twophase_early_simple

        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    def loss_criterion(self, inputs, targets, epoch, earliness_factor):
        """a wrapper around several possible loss functions for experiments"""

        ## try to optimize for earliness only when classification is correct
        if self.lossmode=="early_reward":
            return model.early_loss(inputs,targets,earliness_factor)

        # first cross entropy then early reward loss
        elif self.lossmode == "twophase_early_reward":
            if epoch < self.switch_epoch:
                return self.model.loss_cross_entropy(inputs, targets)
            else:
                return self.model.early_loss_simple(inputs, targets, alpha=earliness_factor)

        # first cross-entropy loss then linear classification loss and simple t/T regularization
        elif self.lossmode=="twophase_linear_loss":
            if epoch < self.switch_epoch:
                return self.model.loss_cross_entropy(inputs, targets)
            else:
                return self.model.early_loss_linear(inputs, targets, alpha=earliness_factor)

        # first cross entropy on all dates, then cross entropy plus simple t/T regularization
        elif self.lossmode == "twophase_early_simple":
            if epoch < self.switch_epoch:
                return self.model.loss_cross_entropy(inputs, targets)
            else:
                return self.model.early_loss_simple(inputs, targets, alpha=earliness_factor)

        else:
            raise ValueError("wrong loss_mode please choose either 'early_reward',  "
                             "'twophase_early_reward', 'twophase_linear_loss', or 'twophase_early_simple'")

    def fit(self,epoch=0):

        for epoch in range(epoch,self.epochs):

            print()

            self.train_epoch(epoch)
            self.test_epoch(epoch)

            self.visdom.plot_epochs(self.logger.get_data())


    def train_epoch(self, epoch):
        self.logger.set_mode("train")

        printer = Printer(prefix="train: ")

        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.nclasses)

        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, targets = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            loss, logprobabilities, weights, stats = self.loss_criterion(inputs, targets, epoch, self.earliness_factor)

            prediction = model.predict(logprobabilities, weights)

            loss.backward()
            self.optimizer.step()

            stats = metric.add(stats)
            stats["accuracy"] = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

        printer.print(stats, iteration, epoch)

        self.logger.log(stats, epoch)

        return stats

    def test_epoch(self, epoch):
        self.logger.set_mode("test")

        printer = Printer(prefix="valid: ")

        # builds a confusion matrix
        #metric_maxvoted = ClassMetric(num_classes=self.nclasses)
        metric = ClassMetric(num_classes=self.nclasses)
        #metric_all_t = ClassMetric(num_classes=self.nclasses)

        with torch.no_grad():
            for iteration, data in enumerate(self.validdataloader):

                inputs, targets = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                loss, logprobabilities, weights, stats = self.loss_criterion(inputs, targets, epoch, self.earliness_factor)

                prediction = model.predict(logprobabilities, weights)

                stats = metric.add(stats)
                stats["accuracy"] = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(),
                                                          prediction.detach().cpu().numpy())

        self.visdom.confusion_matrix(metric.hist)

        n=targets.shape[0]
        for i in range(n):

            classid = targets[i, 0, 0, 0].cpu().numpy()
            classids = targets.unique()

            Y = logprobabilities.exp()[i, :, :, 0, 0].transpose(0,1)
            self.visdom.plot(Y, name="sample {} P(y) (class={})".format(i,classid), fillarea=True, showlegend=True, legend=["class {}".format(c) for c in classids])
            self.visdom.plot(inputs[i, :, 0 ,0 , 0], name="sample {} x (class={})".format(i,classid))
            self.visdom.bar(weights[i, :, 0, 0], name="sample {} P(t) (class={})".format(i, classid))

        printer.print(stats, iteration, epoch)

        self.logger.log(stats, epoch)

        return stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--dataset', type=str, default="Trace", help='UCR Dataset. Will also name the experiment')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=32, help='Batch Size')
    parser.add_argument(
        '-m', '--model', type=str, default="DualOutputRNN", help='Model variant')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=4, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-l', '--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument(
        '-n', '--num_rnn_layers', type=int, default=1, help='number of RNN layers')
    parser.add_argument(
        '-r', '--hidden_dims', type=int, default=32, help='number of RNN hidden dimensions')
    parser.add_argument(
        '--use_batchnorm', action="store_true", help='use batchnorm instead of a bias vector')
    parser.add_argument(
        '--augment_data_noise', type=float, default=0., help='augmentation data noise factor. defaults to 0.')
    parser.add_argument(
        '-a','--earliness_factor', type=float, default=1, help='earliness factor')
    parser.add_argument(
        '-x', '--experiment', type=str, default="", help='experiment prefix')
    parser.add_argument(
        '--loss_mode', type=str, default="twophase_early_simple", help='which loss function to choose. '
                                                                       'valid options are early_reward,  '
                                                                       'twophase_early_reward, '
                                                                       'twophase_linear_loss, or twophase_early_simple')
    parser.add_argument(
        '-s', '--switch_epoch', type=int, default=9999, help='epoch at which to switch the loss function '
                                                             'from classification training to early training')

    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":

    args = parse_args()

    if args.dataset == "synthetic":
        traindataset = SyntheticDataset(num_samples=2000, T=100)
        validdataset = SyntheticDataset(num_samples=1000, T=100)
    else:
        traindataset = UCRDataset(args.dataset, partition="train", ratio=.75, randomstate=2,
                                  augment_data_noise=args.augment_data_noise)
        validdataset = UCRDataset(args.dataset, partition="valid", ratio=.75, randomstate=2)

    nclasses = traindataset.nclasses

    # handles multithreaded batching and shuffling

    np.random.seed(0)
    torch.random.manual_seed(0)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, shuffle=True,
                                                  num_workers=args.workers, pin_memory=True)

    np.random.seed(1)
    torch.random.manual_seed(0)
    validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batchsize, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True)

    if args.model == "DualOutputRNN":
        model = DualOutputRNN(input_dim=1, nclasses=nclasses, hidden_dim=args.hidden_dims,
                              num_rnn_layers=args.num_rnn_layers)
    elif args.model == "AttentionRNN":
        model = AttentionRNN(input_dim=1, nclasses=nclasses, hidden_dim=args.hidden_dims, num_rnn_layers=args.num_rnn_layers,
                             use_batchnorm=args.use_batchnorm)
    else:
        raise ValueError("Invalid Model, Please insert either 'DualOutputRNN' or 'AttentionRNN'")


    if torch.cuda.is_available():
        model = model.cuda()

    visdomenv = "{}_{}_{}".format(args.experiment, args.dataset,args.loss_mode.replace("_","-"))

    config = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        earliness_factor=args.earliness_factor,
        visdomenv=visdomenv,
        switch_epoch=args.switch_epoch,
        loss_mode=args.loss_mode
    )

    trainer = Trainer(model,traindataloader,validdataloader,config=config)

    trainer.fit()

    """
    config = dict(
         batchsize=32,
         workers=4,
         epochs = 4000,
         hidden_dims = 2**10,
         learning_rate = 1e-3,
         earliness_factor=1,
         switch_epoch = 4000,
         num_rnn_layers=3,
         dataset="Trace",
         savepath="/home/marc/tmp/model_r1024_e4k.pth",
         loadpath = None,
         silent = False)

    config = dict(
        batchsize=32,
        workers=0,
        epochs=50,
        hidden_dims=2**4,
        learning_rate=1e-2,
        earliness_factor=1,
        switch_epoch=50,
        num_rnn_layers=1,
        dataset="Trace",
        savepath="/home/marc/tmp/model_r1024_e4k.pth",
        loadpath=None,
        silent=False)
        
        """
