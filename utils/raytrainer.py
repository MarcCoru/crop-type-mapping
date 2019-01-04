from models.DualOutputRNN import DualOutputRNN
from models.conv_shapelets import ConvShapeletModel
from utils.UCR_Dataset import UCRDataset
import torch
from utils.trainer import Trainer
import ray.tune

class RayTrainerDualOutputRNN(ray.tune.Trainable):
    def _setup(self, config):

        traindataset = UCRDataset(config["dataset"],
                                  partition="train",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True,
                                  augment_data_noise=0)

        validdataset = UCRDataset(config["dataset"],
                                  partition="valid",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True)

        nclasses = traindataset.nclasses

        self.epochs = config["epochs"]

        # handles multitxhreaded batching andconfig shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
                                                           num_workers=config["workers"],
                                                           pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
                                                      num_workers=config["workers"], pin_memory=False)

        self.model = DualOutputRNN(input_dim=1,
                                   nclasses=nclasses,
                                   hidden_dims=config["hidden_dims"],
                                   num_rnn_layers=config["num_layers"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train five epochs and then infer once. to avoid overhead on these small datasets
        for i in range(self.epochs):
            self.trainer.train_epoch(epoch=None)

        return self.trainer.test_epoch(epoch=None)

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)

class RayTrainerConv1D(ray.tune.Trainable):
    def _setup(self, config):

        traindataset = UCRDataset(config["dataset"],
                                  partition="train",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True,
                                  augment_data_noise=0)

        validdataset = UCRDataset(config["dataset"],
                                  partition="valid",
                                  ratio=.8,
                                  randomstate=config["fold"],
                                  silent=True)

        self.epochs = config["epochs"]

        nclasses = traindataset.nclasses

        # handles multitxhreaded batching andconfig shuffling
        self.traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=config["batchsize"], shuffle=True,
                                                           num_workers=config["workers"],
                                                           pin_memory=False)
        self.validdataloader = torch.utils.data.DataLoader(validdataset, batch_size=config["batchsize"], shuffle=False,
                                                      num_workers=config["workers"], pin_memory=False)

        self.model = ConvShapeletModel(num_layers=config["num_layers"],
                                       hidden_dims=config["hidden_dims"],
                                       ts_dim=1,
                                       n_classes=nclasses,
                                       use_time_as_feature=True,
                                       drop_probability=config["drop_probability"],
                                       scaleshapeletsize=False,
                                       shapelet_width_increment=config["shapelet_width_increment"])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.trainer = Trainer(self.model, self.traindataloader, self.validdataloader, config)

    def _train(self):
        # epoch is used to distinguish training phases. epoch=None will default to (first) cross entropy phase

        # train epochs and then infer once. to avoid overhead on these small datasets
        for i in range(self.epochs):
            self.trainer.train_epoch(epoch=None)

        return self.trainer.test_epoch(epoch=None)

    def _save(self, path):
        path = path + ".pth"
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
