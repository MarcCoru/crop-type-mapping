import numpy as np
import pandas as pd
import os
import pickle

class Logger():

    def __init__(self, columns, modes, epoch=0, idx=0, rootpath=None, verbose=True):

        self.columns=columns
        self.mode=modes[0]
        self.epoch=epoch
        self.idx = idx
        self.data = pd.DataFrame(columns=["epoch","iteration","mode"]+self.columns)
        self.stored_arrays = dict()
        self.rootpath=rootpath
        self.verbose = verbose

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

        path = os.path.join(self.rootpath,"npy")
        #pickle.dump(self, open( path + "/logger.pkl", "wb" ))

        if not os.path.exists(path):
            os.makedirs(path)

        arrayfile = "{name}_{epoch}.npy"
        csvfile = "data.csv"

        for k,v in self.stored_arrays.items():
            for epoch, data in v:
                filepath = os.path.join(path,arrayfile.format(epoch=epoch, name=k))
                np.save(filepath, data)
                if self.verbose: print("saving "+filepath)

        self.data.to_csv(os.path.join(self.rootpath,csvfile))
