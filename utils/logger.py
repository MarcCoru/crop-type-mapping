import numpy as np
import datetime
import pandas as pd

class Printer():

    def __init__(self, batchsize = None, N = None, prefix=""):
        self.batchsize = batchsize
        self.N = N
        self.prefix = prefix

        self.last=datetime.datetime.now()
        self.lastepoch=0

    def print(self, stats, iteration, epoch):
        print_lst = list()

        if self.N is None:
            print_lst.append('Epoch {}: iteration: {}'.format(epoch, iteration))
        else:
            print_lst.append('Epoch {}: iteration: {}/{}'.format(epoch, iteration, self.N))

        for k, v in zip(stats.keys(), stats.values()):
            if not np.isnan(v):
                print_lst.append('{}: {:.2f}'.format(k, v))

        # replace line if epoch has not changed
        if self.lastepoch==epoch:
            print('\r' + self.prefix + ', '.join(print_lst), end="")
        else:
            print("\n" + self.prefix + ', '.join(print_lst), end="")

        self.last = datetime.datetime.now()

        self.lastepoch=epoch


class Logger():

    def __init__(self, columns, modes, csv=None, epoch=0, idx=0):

        self.columns=columns
        self.mode=modes[0]
        self.epoch=epoch
        self.idx = idx
        self.data = pd.DataFrame(columns=["epoch","iteration","mode"]+self.columns)
        self.csv = csv

    def resume(self, data):
        self.data = data
        self.idx = data.index[-1]
        self.epoch = data["epoch"].max()

    def update_epoch(self, epoch=None):
        if epoch is None:
            self.epoch+=1
        else:
            self.epoch=epoch

    def set_mode(self,mode):
        self.mode = mode

    def log(self, stats, epoch):

        stats["epoch"] = epoch
        stats["mode"] = self.mode

        row = pd.DataFrame(stats, index=[self.idx])

        self.data = self.data.append(row, sort=False)
        self.idx +=1

    def get_data(self):
        return self.data

    def save_csv(self, path=None):
        if path is not None:
            self.data.to_csv(path)
        elif self.csv is not None:
            self.data.to_csv(self.csv)
        else:
            raise ValueError("please provide either path argument or initialize Logger() with csv argument")
