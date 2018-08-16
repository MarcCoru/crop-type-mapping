from __future__ import division
import tensorflow as tf
import numpy as np

from tslearn.datasets import CachedDatasets
from keras.utils import to_categorical

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class DualOutputRNN:
    def __init__(self, n_classes,
                 earliness_factor=1.,
                 lstm_size=128,
                 epochs=100,
                 lr=.01,
                 batch_size=128,
                 reg=1.,
                 ts_size=200,
                 ts_dim=1):
        self.n_classes = n_classes
        self.earliness_factor = earliness_factor
        self.lstm_size = lstm_size
        self.ts_dim = ts_dim
        self.ts_size = ts_size

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg = reg

        self.time_series = None
        self.targets = None
        self.loss = None
        self.train_op = None
        self._build_model()

    def _get_loss_earliness(self, t):
        return self.earliness_factor * t

    def _build_model(self):
        self.time_series = tf.placeholder(tf.float32, [self.batch_size, self.ts_size, self.ts_dim])
        self.targets = tf.placeholder(tf.float32, [self.batch_size, self.n_classes])
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # Classifier part (TODO: a bit of an overkill if n_classes=2)
        w_out_classif = tf.Variable(tf.random_normal([self.lstm_size, self.n_classes]))
        b_out_classif = tf.Variable(tf.random_normal([self.n_classes]))

        # Decision-making part (decision about whether to classify or not)
        w_out_dec = tf.Variable(tf.random_normal([self.lstm_size, 1]))
        b_out_dec = tf.Variable(tf.random_normal([1]))

        # Initialization
        hidden_state = tf.zeros([self.batch_size, self.lstm_size])
        current_state = tf.zeros([self.batch_size, self.lstm_size])
        state = hidden_state, current_state

        self.loss = 0.0
        unstacked_time_series = tf.unstack(self.time_series, self.ts_size, 1)
        self.sum_Pt = [0.0]
        proba_not_decided_yet = [1.0]
        for t in range(self.ts_size):
            # Model
            output, state = lstm(unstacked_time_series[t], state)
            logits_class = tf.matmul(output, w_out_classif) + b_out_classif
            proba_dec = tf.sigmoid(tf.matmul(output, w_out_dec) + b_out_dec)

            # Probabilities
            Pt = proba_dec * proba_not_decided_yet[-1]
            self.sum_Pt.append(self.sum_Pt[-1] + Pt)
            proba_not_decided_yet.append(proba_not_decided_yet[-1] * (1.0 - proba_dec))

            # Loss
            loss_classif = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits_class, labels=self.targets),
                                      shape=(-1, 1))
            self.loss += Pt * (loss_classif + self._get_loss_earliness(t))
        self.loss += self.reg * (1.0 - self.sum_Pt[-1]) ** 2
        self.loss = tf.reduce_mean(self.loss)  # Mean over batch

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def fit(self, X, y, sess):
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(self.epochs):
            avg_cost = 0.
            n_batches = int(X.shape[0] / self.batch_size)
            for _ in range(n_batches):
                indices = np.random.randint(low=0, high=X.shape[0], size=self.batch_size, )
                batch_x, batch_y = X[indices], y[indices]
                _, c = sess.run([self.train_op, self.loss], feed_dict={self.time_series: batch_x,
                                                                       self.targets: batch_y})
                avg_cost += c / n_batches
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

    def predict(self, X, sess):
        pass  # TODO


if __name__ == "__main__":
    sess = tf.Session()
    X, y, _, _ = CachedDatasets().load_dataset("Trace")
    y = to_categorical(y)
    model = DualOutputRNN(n_classes=y.shape[1], batch_size=10, ts_size=X.shape[1], epochs=50, earliness_factor=.01)
    model.fit(X, y, sess)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    sum_Pt = sess.run(model.sum_Pt[1:], feed_dict={model.time_series: X[:model.batch_size], model.targets: y[:model.batch_size]})
    prev_sum = 0.
    for t in range(len(sum_Pt)):
        Pt = sum_Pt[t] - prev_sum
        print(t, Pt[7])  # PRinting for a single TS for the moment
        prev_sum += Pt
