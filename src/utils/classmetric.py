import numpy as np

class ClassMetric(object):
    def __init__(self, num_classes=2, ignore_index=0):
        self.num_classes = num_classes
        _range = -0.5, num_classes - 0.5
        self.range = np.array((_range, _range), dtype=np.int64)
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

        self.store = dict()

        self.earliness_record = list()

    def _update(self, o, t):
        t = t.flatten()
        o = o.flatten()
        # confusion matrix
        n, _, _ = np.histogram2d(t, o, bins=self.num_classes, range=self.range)
        self.hist += n

    def add(self, stats):
        for key, value in stats.items():

            value = value.data.cpu().numpy()

            if key in self.store.keys():
                self.store[key].append(value)
            else:
                self.store[key] = list([value])

        return dict((k, np.stack(v).mean()) for k, v in self.store.items())

    def update_confmat(self, target, output):
        self._update(output, target)
        return self.accuracy()

    def update_earliness(self,earliness):
        self.earliness_record.append(earliness)
        return np.hstack(self.earliness_record).mean()

    def accuracy(self):
        """
        https: // en.wikipedia.org / wiki / Confusion_matrix
        Calculates over all accuracy and per class classification metrics from confusion matrix
        :param confusion_matrix numpy array [n_classes, n_classes] rows True Classes, columns predicted classes:
        :return overall accuracy
                and per class metrics as list[n_classes]:
        """
        confusion_matrix = self.hist

        if type(confusion_matrix) == list:
            confusion_matrix = np.array(confusion_matrix)

        confusion_matrix = confusion_matrix.astype(float)

        total = np.sum(confusion_matrix)
        n_classes, _ = confusion_matrix.shape
        classes = []
        overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

        # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        N = total
        p0 = np.sum(np.diag(self.hist)) / N
        pc = np.sum(np.sum(self.hist, axis=0) * np.sum(self.hist, axis=1)) / N ** 2
        kappa = (p0 - pc) / (1 - pc)

        recall = np.diag(self.hist) / (np.sum(self.hist, axis=1) + 1e-12)
        precision = np.diag(self.hist) / (np.sum(self.hist, axis=0) + 1e-12)
        f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

        # Per class accuracy
        cl_acc = np.diag(self.hist) / (self.hist.sum(1) + 1e-12)

        return dict(
            overall_accuracy=overall_accuracy,
            kappa=kappa,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=cl_acc
        )

