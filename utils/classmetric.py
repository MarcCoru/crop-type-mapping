import numpy as np

class ClassMetric(object):
    def __init__(self, num_classes=2, ignore_index=0):
        self.num_classes = num_classes
        _range = -0.5, num_classes - 0.5
        self.range = np.array((_range, _range), dtype=np.int64)
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    def _update(self, o, t):
        t = t.flatten()
        o = o.flatten()
        # confusion matrix
        n, _, _ = np.histogram2d(t, o, bins=self.num_classes, range=self.range)
        self.hist += n

    def _metrics(self):
        return calculate_accuracy_metrics(self.hist)

    def __call__(self, target, output):
        self._update(output, target)
        return self._metrics()


def calculate_accuracy_metrics(confusion_matrix):
    """
    https: // en.wikipedia.org / wiki / Confusion_matrix
    Calculates over all accuracy and per class classification metrics from confusion matrix
    :param confusion_matrix numpy array [n_classes, n_classes] rows True Classes, columns predicted classes:
    :return overall accuracy
            and per class metrics as list[n_classes]:
    """
    if type(confusion_matrix) == list:
        confusion_matrix = np.array(confusion_matrix)

    confusion_matrix = confusion_matrix.astype(float)

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    classes = []
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    return {"accuracy":overall_accuracy}

    for c in range(n_classes):
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp
        tn = np.sum(np.diag(confusion_matrix)) - tp
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        if (tp + fn) > 0:
            recall = tp / (tp + fn)  # aka sensitivity, hitrate, true positive rate
        else:
            recall = None

        if (fp + tn) > 0:
            specificity = tn / (fp + tn)  # aka true negative rate
        else:
            specificity = None

        if (tp + fp) > 0:
            precision = tp / (tp + fp)  # aka positive predictive value
        else:
            precision = None

        if (2 * tp + fp + fn) > 0:
            fscore = (2 * tp) / (2 * tp + fp + fn)
        else:
            fscore = None

        # http://epiville.ccnmtl.columbia.edu/popup/how_to_calculate_kappa.html
        probability_observed = np.sum(np.diag(confusion_matrix)) / total
        colsum = np.sum(confusion_matrix[:, c])
        rowsum = np.sum(confusion_matrix[c, :])
        probability_expected = (colsum * rowsum / total ** 2) + (total - colsum) * (total - rowsum) / total ** 2

        kappa = (probability_observed - probability_expected) / (1 - probability_expected)

        if (tp + fp + fn + tn) > 0:
            random_accuracy = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / (tp + fp + fn + tn) ** 2
            kappa = (accuracy - random_accuracy) / (1 - random_accuracy)
        else:
            kappa = random_accuracy = None

        if precision != None and recall != None:
            classes.append([precision, recall, fscore, kappa])
        else:
            return dict()

        precision, recall, fscore, kappa = np.array(classes).mean(axis=0)

        stats = dict(
            overall_accuracy=overall_accuracy,
            precision=precision,
            recall=recall,
            fscore=fscore,
            kappa=kappa
        )

        return stats
