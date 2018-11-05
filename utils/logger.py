import numpy as np
import datetime
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

try:
    from visdom import Visdom
except:
    print("could not find visdom package. try 'pip install visdom'. continue without...")
    Visdom=None
    pass


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

class VisdomLogger():
    def __init__(self,**kwargs):
        if Visdom is None:
            self.viz = None # do nothing
            return

        self.viz = Visdom(**kwargs)
        self.windows = dict()

        r = np.random.RandomState(1)
        self.colors = r.randint(0,255, size=(255,3))
        self.colors[0] = np.array([1., 1., 1.])
        self.colors[1] = np.array([0. , 0.18431373, 0.65490196]) # ikb blue

    def update(self, data):
        self.plot_epochs(data)

    def bar(self,X, name="barplot"):
        """
        :param distribution: t
        """
        win = name.replace(" ","_")

        opts = dict(
            title=name,
            xlabel='t',
            ylabel="P(t)",
            width=600,
            height=200,
            marginleft=20,
            marginright=20,
            marginbottom=20,
            margintop=30
        )

        self.viz.bar(X,win=win,opts=opts)

    def plot(self, X, name="plot"):
        """
        :param distribution: t
        """
        win = "pl_"+name.replace(" ","_")

        opts = dict(
            title=name,
            xlabel='t',
            ylabel="P(t)",
            width=600,
            height=200,
            marginleft=20,
            marginright=20,
            marginbottom=20,
            margintop=30
        )

        self.viz.line(X ,win=win, opts=opts)

    def confusion_matrix(self, cm):
        plt.clf()

        name="confusion matrix"

        plt.rcParams['figure.figsize'] = (6, 6)
        #sn.set(font_scale=1.4)  # for label size
        ax = sn.heatmap(cm, annot=True, annot_kws={"size": 11})  # font size
        ax.set(xlabel='ground truth', ylabel='predicted', title="Confusion Matrix")
        plt.tight_layout()
        opts = dict(
            resizeable=True
        )

        self.viz.matplot(plt, win=name, opts=opts)

        pass


    def plot_epochs(self, data):
        """
        Plots mean of epochs
        :param data:
        :return:
        """
        if self.viz is None:
            return # do nothing

        if not self.viz.check_connection():
            return # do nothing

        data_mean_per_epoch = data.groupby(["mode", "epoch"]).mean()
        cols = data_mean_per_epoch.columns
        modes = data_mean_per_epoch.index.levels[0]

        for name in cols:

             if name in self.windows.keys():
                 win = self.windows[name]
                 update = 'new'
             else:
                 win = name # first log -> new window
                 update = None

             opts = dict(
                 title=name,
                 showlegend=True,
                 xlabel='epochs',
                 ylabel=name)

             for mode in modes:

                 epochs = data_mean_per_epoch[name].loc[mode].index
                 values = data_mean_per_epoch[name].loc[mode]

                 win = self.viz.line(
                     X=epochs,
                     Y=values,
                     name=mode,
                     win=win,
                     opts=opts,
                     update=update
                 )
                 update='insert'

             self.windows[name] = win
