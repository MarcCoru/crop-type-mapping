import torch
from utils.classmetric import ClassMetric
from utils.logger import Printer, VisdomLogger, Logger

class Trainer():

    def __init__(self, model, traindataloader, validdataloader, config):

        self.epochs = config["epochs"]
        learning_rate = config["learning_rate"]
        self.earliness_factor = config["earliness_factor"]
        self.switch_epoch = "switch_epoch" if config["switch_epoch"] in config.keys() else 9999
        self.batch_size = validdataloader.batch_size

        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses

        if "visdomenv" in config.keys():
            self.visdom = VisdomLogger(env=config["visdomenv"])
            self.logger = Logger(columns=["accuracy"], modes=["train", "test"], rootpath=config["store"])
            self.show_n_samples = config["show_n_samples"]

        # early_reward,  twophase_early_reward, twophase_linear_loss, or twophase_early_simple
        self.lossmode = config["loss_mode"] if "loss_mode" in config.keys() else "twophase_linear_loss"

        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    def loss_criterion(self, inputs, targets, epoch, earliness_factor):
        """a wrapper around several possible loss functions for experiments"""
        if epoch is None:
            return self.model.loss_cross_entropy(inputs, targets)

        ## try to optimize for earliness only when classification is correct
        if self.lossmode=="early_reward":
            return self.model.early_loss(inputs,targets,earliness_factor)

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
        printer = Printer()

        for epoch in range(epoch,self.epochs):

            self.logger.set_mode("train")
            stats = self.train_epoch(epoch)
            printer.print(stats, epoch, prefix="train: ")

            self.logger.set_mode("test")
            stats = self.test_epoch(epoch)
            self.logger.log(stats, epoch)
            printer.print(stats, epoch, prefix="valid: ")

            self.visdom.confusion_matrix(stats["confusion_matrix"])

            legend = ["class {}".format(c) for c in range(self.nclasses)]

            targets = stats["targets"]

            # either user-specified value or all available values
            n_samples = self.show_n_samples if self.show_n_samples < targets.shape[0] else targets.shape[0]

            for i in range(n_samples):
                classid = targets[i, 0]

                self.visdom.plot(stats["probas"][:, i, :], name="sample {} P(y) (class={})".format(i, classid), fillarea=True,
                                 showlegend=True, legend=legend)
                self.visdom.plot(stats["inputs"][i, :, 0], name="sample {} x (class={})".format(i, classid))
                self.visdom.bar(stats["weights"][i, :], name="sample {} P(t) (class={})".format(i, classid))

            self.visdom.plot_epochs(self.logger.get_data())

        self.logger.save()

    def train_epoch(self, epoch):
        # sets the model to train mode: dropout is applied
        self.model.train()

        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.nclasses)

        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, targets = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            loss, logprobabilities, weights, stats = self.loss_criterion(inputs, targets, epoch, self.earliness_factor)

            prediction = self.model.predict(logprobabilities, weights)

            loss.backward()
            self.optimizer.step()

            stats = metric.add(stats)
            stats["accuracy"] = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())

        return stats

    def test_epoch(self, epoch):
        # sets the model to train mode: no dropout is applied
        self.model.eval()

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

                prediction = self.model.predict(logprobabilities, weights)

                stats = metric.add(stats)
                stats["accuracy"] = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(),
                                                          prediction.detach().cpu().numpy())

        stats["confusion_matrix"] = metric.hist
        stats["targets"] = targets.cpu().numpy()
        stats["inputs"] = inputs.cpu().numpy()
        stats["weights"] = weights.cpu().numpy()

        probas = logprobabilities.exp().transpose(0, 1)
        stats["probas"] = probas.cpu().numpy()

        return stats
