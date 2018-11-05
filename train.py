import torch
from models.DualOutputRNN import DualOutputRNN
from models.AttentionRNN import AttentionRNN
from utils.UCR_Dataset import UCRDataset
from utils.classmetric import ClassMetric
from utils.logger import Printer, VisdomLogger, Logger
import argparse


class Trainer():

    def __init__(self,model, traindataloader, validdataloader, config):

        self.epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        self.earliness_factor = config["earliness_factor"]

        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses

        self.visdom = VisdomLogger()
        self.logger = Logger(columns=["accuracy"], modes=["train", "test"])

        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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

            loss, prediction, weights, stats = self.model.loss(inputs, targets, alpha=self.earliness_factor)

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

                loss, prediction, weights, stats = self.model.loss(inputs, targets, alpha=self.earliness_factor)

                stats = metric.add(stats)
                stats["accuracy"] = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(),
                                                          prediction.detach().cpu().numpy())


        self.visdom.confusion_matrix(metric.hist)

        for i in range(3):
            self.visdom.plot(inputs[i, :, 0 ,0 , 0], name="input {} (class {})".format(i,targets[i,0,0,0]))
            self.visdom.bar(weights[i, :, 0, 0], name="P(t) sample "+str(i))

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
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    args, _ = parser.parse_known_args()
    return args

if __name__=="__main__":

    args = parse_args()

    traindataset = UCRDataset(args.dataset, partition="train", ratio=.75, randomstate=2,
                              augment_data_noise=args.augment_data_noise)
    validdataset = UCRDataset(args.dataset, partition="valid", ratio=.75, randomstate=2)

    nclasses = traindataset.nclasses

    # handles multitxhreaded batching and shuffling
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batchsize, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
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

    config = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        earliness_factor=args.earliness_factor,
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
