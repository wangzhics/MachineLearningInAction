import numpy as np


class OrdinaryLeastSquares:
    def __init__(self, x_arrays, y_array):
        # train data
        self._x_arrays = x_arrays
        self._y_array = y_array
        # train result
        self._had_regress = False
        self._x_mat = None
        self._x_mat_t = None
        self._y_mat = None

    def regress(self, r_x, k):
        if not self._had_regress:
            self._x_mat = np.mat(self._x_arrays)
            self._x_mat_t = self._x_mat.transpose()
            self._y_mat = np.mat(self._y_array).transpose()
            self._had_regress = True
        m, n = np.shape(self._x_mat)
        weights_eye = np.eye(m)
        r_x_mat = np.mat(r_x)
        for i in range(m):
            differ_mat = r_x_mat - self._x_mat[i, :]
            # LWLR: Locally Weighted Linear Regression
            weights_eye[i, i] = np.exp(differ_mat * differ_mat.transpose() / (-2.0 * np.square(k)))
        tmp_mat = self._x_mat_t * weights_eye * self._x_mat
        tmp_mat_det = np.linalg.det(tmp_mat)
        if tmp_mat_det == 0.0:
            print("the xTx matrix is singular, can not do inverse")
            return
        weights = np.linalg.inv(tmp_mat) * self._x_mat_t * weights_eye * self._y_mat
        # get the value of 1 * 1 ,atrix
        return np.sum(r_x_mat * weights)
