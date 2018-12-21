import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sn
from visdom import Visdom

class Printer():

    def __init__(self, batchsize = None):
        self.batchsize = batchsize

        self.last=datetime.datetime.now()
        self.lastepoch=0

    def print(self, stats, epoch, iteration=None, prefix=""):
        print_lst = list()

        print_lst.append('Epoch {}:'.format(epoch))

        if iteration is not None:
            print_lst.append(" iteration: {}".format(iteration))

        for k, v in zip(stats.keys(), stats.values()):
            if np.array(v).size == 1:
                if not np.isnan(v):
                    print_lst.append('{}: {:.2f}'.format(k, v))

        # replace line if epoch has not changed
        if self.lastepoch==epoch:
            print('\r' + prefix + ', '.join(print_lst), end="")
        else:
            print("\n" + prefix + ', '.join(print_lst), end="")

        self.last = datetime.datetime.now()
        self.lastepoch=epoch

class Logger():

    def __init__(self, columns, modes, epoch=0, idx=0, rootpath=None):

        self.columns=columns
        self.mode=modes[0]
        self.epoch=epoch
        self.idx = idx
        self.data = pd.DataFrame(columns=["epoch","iteration","mode"]+self.columns)
        self.stored_arrays = dict()
        self.rootpath=rootpath

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

        clean_stats = dict()
        for k,v in stats.items():
            if np.array(v).size == 1:
                clean_stats[k] = v
            else:
                self.log_array(name=k,array=v, epoch=epoch)

        self.log_numbers(clean_stats, epoch)

    def log_array(self, name, array, epoch):

        if name not in self.stored_arrays.keys():
            self.stored_arrays[name] = list()

        self.stored_arrays[name].append((epoch, array))

    def log_numbers(self, stats, epoch):

        stats["epoch"] = epoch
        stats["mode"] = self.mode

        row = pd.DataFrame(stats, index=[self.idx])

        self.data = self.data.append(row, sort=False)
        self.idx +=1

    def get_data(self):
        return self.data

    def save(self):

        if not os.path.exists(self.rootpath):
            os.makedirs(self.rootpath)

        arrayfile = "{epoch}_{name}.npy"
        csvfile = "data.csv"

        for k,v in self.stored_arrays.items():
            for el in v:
                epoch, data = el
                filename = arrayfile.format(epoch=epoch, name=k)
                np.save(os.path.join(self.rootpath,filename),data)

        self.data.to_csv(os.path.join(self.rootpath,csvfile))

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
        X[np.isnan(X)] = 0

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

    def plot(self, X, name="plot",**kwargs):

        X[np.isnan(X)] = 0

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
            margintop=30,
            **kwargs
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

    def plot_class_p(self,X):
        plt.clf()

        x = X.detach().cpu().numpy()
        plt.plot(x[0, :])

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