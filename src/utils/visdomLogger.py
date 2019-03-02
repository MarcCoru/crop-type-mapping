import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from visdom import Visdom

class VisdomLogger():
    def __init__(self,**kwargs):

        if Visdom is None:
            self.viz = None # do nothing
            return

        self.connected = True
        try:
            self.viz = Visdom(raise_exceptions=True,**kwargs)
        except Exception as e:
            print("Could not reach visdom server...")
            self.connected = False
            pass

        self.windows = dict()

        r = np.random.RandomState(1)
        self.colors = r.randint(0,255, size=(255,3))
        self.colors[0] = np.array([1., 1., 1.])
        self.colors[1] = np.array([0. , 0.18431373, 0.65490196]) # ikb blue

    def update(self, data):
        if self.connected:
            self.plot_epochs(data)

    def bar(self,X, name="barplot"):
        if self.connected:
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
        if self.connected:

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

    def confusion_matrix(self, cm, title="Confusion Matrix", norm=None):
        if self.connected:
            plt.clf()

            if norm is not None:
                cm /= np.expand_dims(cm.sum(norm),axis=norm)
                cm[np.isnan(cm)] = 0
                cm[np.isinf(cm)] = 0
                vmin = 0
                vmax = 1
            else:
                vmin = None
                vmax = None

            name=title

            plt.rcParams['figure.figsize'] = (9, 9)
            #sn.set(font_scale=1.4)  # for label size
            ax = sn.heatmap(cm, annot=True, annot_kws={"size": 11}, vmin=vmin, vmax=vmax)  # font size
            ax.set(xlabel='ground truth', ylabel='predicted', title=title)
            plt.tight_layout()
            opts = dict(
                resizeable=True
            )

            self.viz.matplot(plt, win=name, opts=opts)

    def plot_class_p(self,X):
        if self.connected:
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

    def plot_boxplot(self, labels, t_stops, tmin=None, tmax=None):

        if self.connected:
            grouped = [t_stops[labels == i] for i in np.unique(labels)]
            #legend = ["class {}".format(i) for i in np.unique(labels)]

            plt.clf()

            name = "boxplot"

            plt.rcParams['figure.figsize'] = (9, 9)
            # sn.set(font_scale=1.4)  # for label size
            ax = sn.boxplot(data=grouped, orient="h")
            ax.set_xlabel("t_stop")
            ax.set_ylabel("class")
            ax.set_xlim(tmin, tmax)
            #ax = sn.heatmap(cm, annot=True, annot_kws={"size": 11}, vmin=vmin, vmax=vmax)  # font size
            #ax.set(xlabel='ground truth', ylabel='predicted', title=title)
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
        if self.connected:

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