import numpy as np
from pandas import DataFrame
from enum import Enum

class TrainAlgorithm(Enum):
    GradientAscent = 1
    RandomGradientAscent = 2
    ImproveGradientAscent = 3


class LogisticRegression:
    def __init__(self):
        # train data
        self._train_frame = DataFrame({"feature": [], "label": []})
        # train result
        self._weights = np.ones(0)

    def set_train_frame(self, train_frame):
        self._train_frame["feature"] = train_frame["feature"]
        self._train_frame["label"] = train_frame["label"]

    def add_train_features(self, train_features, train_labels):
        for i in range(len(train_features)):
            self._train_frame = \
                self._train_frame.append({"feature": train_features[i], "label": train_labels[i]}, ignore_index=True)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def _train_with_ga(self):
        feature_mat = np.mat(self._train_frame["feature"].tolist())
        label_mat = np.mat(self._train_frame["label"].tolist()).transpose()
        m, n = np.shape(feature_mat)
        alpha = 0.001
        weights = np.ones((n, 1))
        for k in range(500):
            h = self._sigmoid(feature_mat * weights)
            e = (label_mat - h)
            weights = weights + alpha * feature_mat.transpose() * e
        self._weights = weights

    def _train_with_rga(self):
        pass

    def _train_wth_iga(self):
        pass

    def train(self, type):
        if type == TrainAlgorithm.GradientAscent:
            self._train_with_ga()
        elif type == TrainAlgorithm.RandomGradientAscent:
            self._train_with_rga()
        elif type == TrainAlgorithm.ImproveGradientAscent:
            self._train_wth_iga()
        else:
            self._train_wth_iga()
        print(self._weights)

    def get_weights(self):
        return self._weights