import numpy as np
import datetime
import pandas as pd

class Printer():

    def __init__(self, batchsize = None, N = None):
        self.batchsize = batchsize
        self.N = N

        self.last=datetime.datetime.now()
        self.lastepoch=0

    def print(self, stats, iteration, epoch):
        print_lst = list()

        if self.N is None:
            print_lst.append('Epoch {}: iteration: {}'.format(epoch, iteration))
        else:
            print_lst.append('Epoch {}: iteration: {}/{}'.format(epoch, iteration, self.N))

        dt = (datetime.datetime.now() - self.last).total_seconds()

        print_lst.append('logs/sec: {:.2f}'.format(dt / 1))

        if self.batchsize is not None:
            print_lst.append('samples/sec: {:.2f}'.format(dt / self.batchsize))

        for k, v in zip(stats.keys(), stats.values()):
            if not np.isnan(v):
                print_lst.append('{}: {:.2f}'.format(k, v))

        # replace line if epoch has not changed
        if self.lastepoch==epoch:
            print('\r' + ', '.join(print_lst), end="")
        else:
            print("\n"+', '.join(print_lst), end="")

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
