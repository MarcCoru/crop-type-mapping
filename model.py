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
        self.init_hidden_state = None
        self.init_current_state = None
        self.predicted_probas = None
        self.predicted_decs = None
        self.loss = None
        self.loss_no_early = None
        self.train_op = None
        self.train_op_no_early = None
        self._build_model()

    def _get_loss_earliness(self, t):
        return self.earliness_factor * t

    def _build_model(self):
        self.time_series = tf.placeholder(tf.float32, [None, self.ts_size, self.ts_dim])
        self.targets = tf.placeholder(tf.float32, [None, self.n_classes])
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # Classifier part
        if self.n_classes == 2:
            w_out_classif = tf.Variable(tf.random_normal([self.lstm_size, 1]))
            b_out_classif = tf.Variable(tf.random_normal([1]))
        else:
            w_out_classif = tf.Variable(tf.random_normal([self.lstm_size, self.n_classes]))
            b_out_classif = tf.Variable(tf.random_normal([self.n_classes]))

        # Decision-making part (decision about whether to classify or not)
        w_out_dec = tf.Variable(tf.random_normal([self.lstm_size, 1]))
        b_out_dec = tf.Variable(tf.random_normal([1]))

        # Initialization
        self.init_hidden_state = tf.placeholder(tf.float32, [None, self.lstm_size])
        self.init_current_state = tf.placeholder(tf.float32, [None, self.lstm_size])
        state = self.init_hidden_state, self.init_current_state

        self.loss = 0.0
        self.loss_no_early = 0.0
        unstacked_time_series = tf.unstack(self.time_series, self.ts_size, 1)
        sum_Pt = [0.0]
        self.predicted_probas = []
        self.predicted_decs = []
        proba_not_decided_yet = [1.0]
        for t in range(self.ts_size):
            # Model
            output, state = lstm(unstacked_time_series[t], state)
            logits_class = tf.matmul(output, w_out_classif) + b_out_classif
            self.predicted_probas.append(tf.nn.softmax(logits_class))
            proba_dec = tf.sigmoid(tf.matmul(output, w_out_dec) + b_out_dec)
            self.predicted_decs.append(proba_dec)

            # Probabilities
            if t < self.ts_size - 1:
                Pt = proba_dec * proba_not_decided_yet[-1]
                proba_not_decided_yet.append(proba_not_decided_yet[-1] * (1.0 - proba_dec))
            else:
                Pt = proba_not_decided_yet[-1]
                proba_not_decided_yet.append(0.)
            sum_Pt.append(sum_Pt[-1] + Pt)

            # Loss
            if self.n_classes == 2:
                loss_classif = tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_class,
                                                                                  labels=self.targets),
                                          shape=(-1, 1))
            else:
                loss_classif = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits=logits_class,
                                                                                  labels=self.targets),
                                          shape=(-1, 1))
            self.loss_no_early += tf.reduce_mean(loss_classif) / self.ts_size
            self.loss += Pt * (loss_classif + self._get_loss_earliness(t))
        # self.loss += self.reg * (1.0 - self.sum_Pt[-1]) ** 2
        self.loss = tf.reduce_mean(self.loss)  # Mean over batch

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.train_op_no_early = optimizer.minimize(self.loss_no_early)

    def fit(self, X, y, sess):
        n_epochs_no_early = 100

        init = tf.global_variables_initializer()
        sess.run(init)

        init_states = np.zeros((self.batch_size, self.lstm_size))

        for epoch in range(self.epochs):
            avg_cost = 0.
            n_batches = int(X.shape[0] / self.batch_size)
            for _ in range(n_batches):
                indices = np.random.randint(low=0, high=X.shape[0], size=self.batch_size, )
                batch_x, batch_y = X[indices], y[indices]
                if epoch < n_epochs_no_early:
                    _, c = sess.run([self.train_op_no_early, self.loss_no_early],
                                    feed_dict={self.time_series: batch_x,
                                               self.targets: batch_y,
                                               self.init_hidden_state: init_states,
                                               self.init_current_state: init_states})
                else:
                    _, c = sess.run([self.train_op, self.loss],
                                    feed_dict={self.time_series: batch_x,
                                               self.targets: batch_y,
                                               self.init_hidden_state: init_states,
                                               self.init_current_state: init_states})
                avg_cost += c / n_batches
            if epoch == n_epochs_no_early - 1:
                print("[End of no-early training] Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
            elif epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

    def predict(self, X, sess):
        n_ts = X.shape[0]
        init_states = np.zeros((n_ts, self.lstm_size))
        predicted_probas, predicted_decs = sess.run([self.predicted_probas, self.predicted_decs],
                                                    feed_dict={self.time_series: X,
                                                               self.init_hidden_state: init_states,
                                                               self.init_current_state: init_states})
        y = np.zeros((n_ts, ), dtype=np.int32)
        tau = np.zeros((n_ts, ), dtype=np.int32)
        already_predicted = np.zeros((n_ts, ), dtype=np.bool)

        for t in range(len(predicted_probas)):
            if t < self.ts_size - 1:
                dec_t = predicted_decs[t].reshape((-1, ))
            else:
                dec_t = np.ones((n_ts, ))
            class_indices = predicted_probas[t].argmax(axis=1)
            draws = np.random.random_sample(size=n_ts)
            assign_class_to_y = np.logical_and(draws < dec_t, np.logical_not(already_predicted))

            y[assign_class_to_y] = class_indices[assign_class_to_y]
            tau[assign_class_to_y] = t
            already_predicted[assign_class_to_y] = True
        assert(np.alltrue(already_predicted))
        return y, tau


if __name__ == "__main__":
    sess = tf.Session()
    tf.set_random_seed(0)
    np.random.seed(0)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    y_train = to_categorical(y_train)

    model = DualOutputRNN(n_classes=y_train.shape[1],
                          batch_size=10,
                          ts_size=X_train.shape[1],
                          epochs=200,
                          earliness_factor=.01,
                          lr=.001,
                          reg=.01)
    model.fit(X_train, y_train, sess)

    y_pred, tau_pred = model.predict(X_test, sess)
    for yi, yi_hat, taui in zip(y_test, y_pred, tau_pred):
        print(yi, yi_hat, taui)

