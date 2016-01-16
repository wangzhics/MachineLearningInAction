import numpy as np
from pandas import DataFrame


class OrdinaryLeastSquares:
    def __init__(self):
        # train data
        self._train_frame = DataFrame({"feature": [], "label": []})
        # train result
        self._had_regress = False
        self._feature_mat = None
        self._feature_mat_t = None
        self._label_mat = None

    def set_train_frame(self, train_frame):
        self._train_frame["feature"] = train_frame["feature"]
        self._train_frame["label"] = train_frame["label"]

    def add_train_features(self, train_features, train_labels):
        for i in range(len(train_features)):
            self._train_frame = \
                self._train_frame.append({"feature": train_features[i], "label": train_labels[i]}, ignore_index=True)
        self._had_regress = False

    def regress(self, feature, k):
        if not self._had_regress:
            self._feature_mat = np.mat(self._train_frame["feature"].tolist())
            self._feature_mat_t = self._feature_mat.transpose()
            self._label_mat = np.mat(self._train_frame["label"].tolist()).transpose()
            self._had_regress = True
        m, n = np.shape(self._feature_mat)
        weights_eye = np.eye(m)
        feature_mat = np.mat(feature)
        for i in range(m):
            differ_mat = feature_mat - self._feature_mat[i, :]
            # LWLR: Locally Weighted Linear Regression
            weights_eye[i, i] = np.exp(differ_mat * differ_mat.transpose() / (-2.0 * np.square(k)))
        tmp_mat = self._feature_mat_t * weights_eye * self._feature_mat
        tmp_mat_det = np.linalg.det(tmp_mat)
        if tmp_mat_det == 0.0:
            print("the xTx matrix is singular, can not do inverse")
            return
        weights = np.linalg.inv(tmp_mat) * self._feature_mat_t * weights_eye * self._label_mat
        # get the value of 1 * 1 ,atrix
        return np.sum(feature_mat * weights)
