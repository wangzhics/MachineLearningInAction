import numpy as np


class RidgeRegress:
    def __init__(self, x_arrays, y_array):
        # train data
        self._x_arrays = x_arrays
        self._y_array = y_array
        # standardization matrix
        self._s_calc = False
        self._s_x_mat = None
        self._s_y_mat = None
        # train result
        self._weights = None

    def _calc_standard_matrix(self):
        x_mat = np.mat(self._x_arrays)
        y_mat = np.mat(self._y_array).transpose()
        m, n = np.shape(x_mat)
        # standardization x_mat
        x_mean_mat = x_mat.mean(0)
        x_var_mat = np.var(x_mat, 0)  # variance
        self._s_x_mat = (x_mat - x_mean_mat) / x_var_mat
        # standardization y_mat
        y_mean_mat = y_mat.mean(0)
        self._s_y_mat = y_mat - y_mean_mat

    def train(self, l):
        if self._s_calc is False:
            self._calc_standard_matrix()
            self._s_calc = True
        m, n = np.shape(self._s_x_mat)
        tmp_mat = self._s_x_mat.transpose() * self._s_x_mat + np.eye(n) * np.exp(l)
        if np.linalg.det(tmp_mat) == 0.0:
            print("the xTx matrix is singular, can not do inverse")
            return
        weight_vector = np.linalg.inv(tmp_mat) * self._s_x_mat.transpose() * self._s_y_mat
        self._weights = weight_vector.tolist()

    def get_weights(self):
        return self._weights
