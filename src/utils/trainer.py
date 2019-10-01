import torch
from utils.classmetric import ClassMetric
from sklearn.metrics import roc_auc_score, auc
from utils.printer import Printer

import sys
sys.path.append("../models")
sys.path.append("../models/transformer")
sys.path.append("../models/transformer/TransformerEncoder")
sys.path.append("models")
sys.path.append("..")

import os
import numpy as np
from models.ClassificationModel import ClassificationModel
import torch.nn.functional as F
from utils.scheduled_optimizer import ScheduledOptim
import copy

CLASSIFICATION_PHASE_NAME="classification"
EARLINESS_PHASE_NAME="earliness"

class Trainer():

    def __init__(self,
                 model,
                 traindataloader,
                 validdataloader,
                 epochs=4,
                 learning_rate=0.1,
                 store="/tmp",
                 test_every_n_epochs=1,
                 checkpoint_every_n_epochs=5,
                 visdomlogger=None,
                 optimizer=None,
                 show_n_samples=1,
                 overwrite=True,
                 logger=None,
                 **kwargs):

        self.epochs = epochs
        self.batch_size = validdataloader.batch_size
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses
        self.store = store
        self.test_every_n_epochs = test_every_n_epochs
        self.logger = logger
        self.show_n_samples = show_n_samples
        self.model = model
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.early_stopping_smooth_period = 10
        self.early_stopping_patience = 5
        self.not_improved_epochs=0
        self.early_stopping_metric="kappa"

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        #self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        self.classweights = torch.FloatTensor(traindataloader.dataset.classweights)

        if torch.cuda.is_available():
            self.classweights = self.classweights.cuda()

        if visdomlogger is not None:
            self.visdom = visdomlogger
        else:
            self.visdom = None

        # only save checkpoint if not previously resumed from it
        self.resumed_run = False

        self.epoch = 0

        if os.path.exists(self.get_model_name()) and not overwrite:
            print("Resuming from snapshot {}.".format(self.get_model_name()))
            self.resume(self.get_model_name())
            self.resumed_run = True

    def resume(self, filename):
        snapshot = self.model.load(filename)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.epoch = snapshot["epoch"]
        print("resuming optimizer state")
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.logger.resume(snapshot["logged_data"])

    def snapshot(self, filename):
        self.model.save(
        filename,
        optimizer_state_dict=self.optimizer.state_dict(),
        epoch=self.epoch,
        logged_data=self.logger.get_data())

    def fit(self):
        printer = Printer()

        while self.epoch < self.epochs:
            self.new_epoch() # increments self.epoch

            self.logger.set_mode("train")
            stats = self.train_epoch(self.epoch)
            self.logger.log(stats, self.epoch)
            printer.print(stats, self.epoch, prefix="\n"+self.traindataloader.dataset.partition+": ")

            if self.epoch % self.test_every_n_epochs == 0 or self.epoch==1:
                self.logger.set_mode("test")
                stats = self.test_epoch(self.validdataloader)
                self.logger.log(stats, self.epoch)
                printer.print(stats, self.epoch, prefix="\n"+self.validdataloader.dataset.partition+": ")
                if self.visdom is not None:
                    self.visdom_log_test_run(stats)

            if self.visdom is not None:
                self.visdom.plot_epochs(self.logger.get_data())

            if self.checkpoint_every_n_epochs % self. epoch==0:
                print("Saving model to {}".format(self.get_model_name()))
                self.snapshot(self.get_model_name())
                print("Saving log to {}".format(self.get_log_name()))
                self.logger.get_data().to_csv(self.get_log_name())

            if self.check_for_early_stopping(smooth_period=self.early_stopping_smooth_period):
                print()
                print(f"Model did not improve in the last {self.early_stopping_grace_period} epochs. stopping training...")
                print("Saving model to {}".format(self.get_model_name()))
                self.snapshot(self.get_model_name())
                print("Saving log to {}".format(self.get_log_name()))
                self.logger.get_data().to_csv(self.get_log_name())
                break

        return self.logger

    def check_for_early_stopping(self,smooth_period):
        log = self.logger.get_data()
        log = log.loc[log["mode"] == "test"]

        early_stopping_condition = log[self.early_stopping_metric].diff()[-smooth_period:].mean() < 0 and self.epoch > smooth_period

        if early_stopping_condition:
            self.not_improved_epochs += 1
            print()
            print(f"model did not improve: {self.not_improved_epochs} of {self.early_stopping_patience} until early stopping...")
            return self.not_improved_epochs >= self.early_stopping_patience
        else:
            self.not_improved_epochs = 0
            return False


    def new_epoch(self):
        self.epoch += 1

    def visdom_log_test_run(self, stats):

        # prevent side effects <- normalization of confusion matrix
        stats = copy.deepcopy(stats)

        if "t_stops" in stats.keys(): self.visdom.plot_boxplot(labels=stats["labels"], t_stops=stats["t_stops"], tmin=0, tmax=self.traindataloader.dataset.samplet)

        # if any prefixed "class_" keys are stored
        if np.array(["class_" in k for k in stats.keys()]).any():
            self.visdom.plot_class_accuracies(stats)

        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=None, title="Confusion Matrix", logscale=None)
        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=0, title="Recall")
        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=1, title="Precision")
        legend = ["class {}".format(c) for c in range(self.nclasses)]
        targets = stats["targets"]
        # either user-specified value or all available values
        n_samples = self.show_n_samples if self.show_n_samples < targets.shape[0] else targets.shape[0]

        for i in range(n_samples):
            classid = targets[i, 0]

            if len(stats["probas"].shape) == 3:
                self.visdom.plot(stats["probas"][:, i, :], name="sample {} P(y) (class={})".format(i, classid),
                                 fillarea=True,
                                 showlegend=True, legend=legend)
            self.visdom.plot(stats["inputs"][i, :, 0], name="sample {} x (class={})".format(i, classid))
            if "pts" in stats.keys(): self.visdom.bar(stats["pts"][i, :], name="sample {} P(t) (class={})".format(i, classid))
            if "deltas" in stats.keys(): self.visdom.bar(stats["deltas"][i, :], name="sample {} deltas (class={})".format(i, classid))
            if "budget" in stats.keys(): self.visdom.bar(stats["budget"][i, :], name="sample {} budget (class={})".format(i, classid))

    def get_model_name(self):
        return os.path.join(self.store, "model.pth")

    def get_log_name(self):
        return os.path.join(self.store, "log.csv")

    def train_epoch(self, epoch):
        # sets the model to train mode: dropout is applied
        self.model.train()

        # builds a confusion matrix
        metric = ClassMetric(num_classes=self.nclasses)

        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, targets, _ = data

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1,2))

            loss = F.nll_loss(logprobabilities, targets[:, 0])

            stats = dict(
                loss=loss,
            )

            loss.backward()
            if isinstance(self.optimizer,ScheduledOptim):
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()

            prediction = self.model.predict(logprobabilities)
            t_stop = None

            stats = metric.add(stats)

            accuracy_metrics = metric.update_confmat(targets.mode(1)[0].detach().cpu().numpy(), prediction.detach().cpu().numpy())
            stats["accuracy"] = accuracy_metrics["overall_accuracy"]
            stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()
            stats["mean_recall"] = accuracy_metrics["recall"].mean()
            stats["mean_precision"] = accuracy_metrics["precision"].mean()
            stats["mean_f1"] = accuracy_metrics["f1"].mean()
            stats["kappa"] = accuracy_metrics["kappa"]
            if t_stop is not None:
                earliness = (t_stop.float()/(inputs.shape[1]-1)).mean()
                stats["earliness"] = metric.update_earliness(earliness.cpu().detach().numpy())

        return stats

    def test_epoch(self, dataloader, epoch=None):
        # sets the model to train mode: no dropout is applied
        self.model.eval()

        # builds a confusion matrix
        #metric_maxvoted = ClassMetric(num_classes=self.nclasses)
        metric = ClassMetric(num_classes=self.nclasses)
        #metric_all_t = ClassMetric(num_classes=self.nclasses)

        tstops = list()
        predictions = list()
        probas = list()
        ids_list = list()
        labels = list()


        with torch.no_grad():
            for iteration, data in enumerate(dataloader):

                inputs, targets, ids = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                logprobabilities, deltas, pts, budget = self.model.forward(inputs.transpose(1, 2))

                loss = F.nll_loss(logprobabilities, targets[:, 0])

                stats = dict(
                    loss=loss,
                )

                prediction = self.model.predict(logprobabilities)
                t_stop = None

                ## enter numpy world
                prediction = prediction.detach().cpu().numpy()
                label = targets.mode(1)[0].detach().cpu().numpy()
                if t_stop is not None: t_stop = t_stop.cpu().detach().numpy()
                if pts is not None: pts = pts.detach().cpu().numpy()
                if deltas is not None: deltas = deltas.detach().cpu().numpy()
                if budget is not None: budget = budget.detach().cpu().numpy()

                if t_stop is not None: tstops.append(t_stop)
                predictions.append(prediction)
                labels.append(label)
                probas.append(logprobabilities.exp().detach().cpu().numpy())
                ids_list.append(ids.detach().cpu().numpy())

                stats = metric.add(stats)

                accuracy_metrics = metric.update_confmat(label,
                                                         prediction)

                stats["accuracy"] = accuracy_metrics["overall_accuracy"]
                stats["mean_accuracy"] = accuracy_metrics["accuracy"].mean()

                #for cl in range(len(accuracy_metrics["accuracy"])):
                #    acc = accuracy_metrics["accuracy"][cl]
                #    stats["class_{}_accuracy".format(cl)] = acc

                stats["mean_recall"] = accuracy_metrics["recall"].mean()
                stats["mean_precision"] = accuracy_metrics["precision"].mean()
                stats["mean_f1"] = accuracy_metrics["f1"].mean()
                stats["kappa"] = accuracy_metrics["kappa"]
                if t_stop is not None:
                    earliness = (t_stop.astype(float) / (inputs.shape[1] - 1)).mean()
                    stats["earliness"] = metric.update_earliness(earliness)

            stats["confusion_matrix"] = copy.copy(metric.hist)
            stats["targets"] = targets.cpu().numpy()
            stats["inputs"] = inputs.cpu().numpy()
            if deltas is not None: stats["deltas"] = deltas
            if pts is not None: stats["pts"] = pts
            if budget is not None: stats["budget"] = budget




        if t_stop is not None: stats["t_stops"] = np.hstack(tstops)
        stats["predictions"] = np.hstack(predictions) # N
        stats["labels"] = np.hstack(labels) # N
        stats["probas"] = np.vstack(probas) # NxC
        stats["ids"] = np.hstack(ids_list)

        return stats
