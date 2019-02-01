import numpy as np
import datetime

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