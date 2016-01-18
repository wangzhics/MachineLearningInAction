import numpy as np
import random


class RidgeRegress:
    def __init__(self, x_arrays, y_array):
        # train data
        self._x_arrays = x_arrays
        self._y_array = y_array
        # standardization matrix
        self._s_calc = False
        self._s_x_mat = None
        self._s_y_mat = None
        # variance of x matrix
        self._x_var_mat = None
        # mean of y matrix
        self._x_mean_mat = None
        self._y_mean_mat = None
        # cross train result
        self._weights = None

    def _calc_standard_matrix(self):
        x_mat = np.mat(self._x_arrays)
        y_mat = np.mat(self._y_array).transpose()
        m, n = np.shape(x_mat)
        # standardization x_mat
        self._x_mean_mat = x_mat.mean(0)
        self._x_var_mat = np.var(x_mat, 0)  # variance
        self._s_x_mat = (x_mat - self._x_mean_mat) / self._x_var_mat
        # standardization y_mat
        self._y_mean_mat = y_mat.mean(0)
        self._s_y_mat = y_mat - self._y_mean_mat

    def calc_ridge_weight(self, l):
        if self._s_calc is False:
            self._calc_standard_matrix()
            self._s_calc = True
        m, n = np.shape(self._s_x_mat)
        tmp_mat = self._s_x_mat.transpose() * self._s_x_mat + np.eye(n) * np.exp(l)
        if np.linalg.det(tmp_mat) == 0.0:
            print("the xTx matrix is singular, can not do inverse")
            return
        weight_vector = np.linalg.inv(tmp_mat) * self._s_x_mat.transpose() * self._s_y_mat
        return weight_vector.tolist()

    def cross_train(self):
        best_ridge_lambda = self._get_ridge_lambda()
        best_ridge_weight = self.calc_ridge_weight(best_ridge_lambda)
        # calculate weights
        part_weight_mat = np.mat(best_ridge_weight).transpose() / self._x_var_mat
        constant_mat = -1 * np.sum(np.multiply(self._x_mean_mat, part_weight_mat)) + self._y_mean_mat
        # build train weight
        self._weights = []
        self._weights.extend(constant_mat.tolist())
        part_weight_list =  part_weight_mat.tolist()[0]
        for i in range(len(part_weight_list)):
           self._weights.append([part_weight_list[i]])

    def get_weights(self):
        return self._weights

    def _get_ridge_lambda(self):
        total_row = len(self._x_arrays)
        lambda_list = [i for i in range(-10, 20)]
        variance_mat = np.zeros((10, 30))
        # validate ridge weight 10 times
        for i in range(10):
            # random select 90% data as training data, 10% as validation data
            t_x_arrays = []
            t_y_array = []
            v_x_arrays = []
            v_y_array = []
            random_index_list = [i for i in range(total_row)]
            random.shuffle(random_index_list)
            for j in range(total_row):
                # as training data
                if j < 0.9 * total_row:
                    t_x_arrays.append(self._x_arrays[random_index_list[j]])
                    t_y_array.append(self._y_array[random_index_list[j]])
                # as validation data
                else:
                    v_x_arrays.append(self._x_arrays[random_index_list[j]])
                    v_y_array.append(self._y_array[random_index_list[j]])
            # get validation data matrix
            v_x_mat = np.mat(v_x_arrays)
            v_y_mat = np.mat(v_y_array).transpose()
            # get ridge weight
            ridge = RidgeRegress(t_x_arrays, t_y_array)
            wights_list = []
            k = 0
            for l in lambda_list:
                weight = ridge.calc_ridge_weight(l)
                weight_mat = np.mat(weight)
                variance = ridge._calc_variance(v_x_mat, v_y_mat, weight_mat)
                variance_mat[i][k] = variance
                k += 1
        # get min variance of lamda
        variance_mean_mat = variance_mat.mean(0)
        min_mean = variance_mean_mat.min()
        lambda_index = 0
        for i in range(30):
            if variance_mean_mat[i] == min_mean:
                lambda_index = i
        lambda_value = lambda_list[lambda_index]
        return lambda_value

    def _calc_variance(self, v_x_mat, v_y_mat, weight_mat):
        v_x_s_mat = (v_x_mat - self._x_mean_mat) / self._x_var_mat
        v_y_tmp_mat = v_x_s_mat * weight_mat + self._y_mean_mat
        variance = np.square(v_y_tmp_mat - v_y_mat).sum()
        return variance




