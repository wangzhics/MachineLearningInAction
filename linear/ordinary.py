import numpy as np
from pandas import DataFrame


class LeastSquares:
    def __init__(self, x_arrays, y_array):
        # train data
        self._x_arrays = x_arrays
        self._y_array = y_array
        # train result
        self._weights = None

    def train(self):
        x_mat = np.mat(self._x_arrays)
        y_mat = np.mat(self._y_array).transpose()
        tmp_mat = x_mat.transpose() * x_mat
        if np.linalg.det(tmp_mat) == 0.0:
            print("the xTx matrix is singular, can not do inverse")
            return
        self._weights = np.linalg.inv(tmp_mat) * x_mat.transpose() * y_mat

    def get_weights(self):
        return self._weights.tolist()

    def regress(self, x):
        x_mat = np.mat(x)
        y_mat = x_mat * self._weights
        return np.sum(y_mat)
