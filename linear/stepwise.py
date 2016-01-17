from linear.ridge import RidgeRegress
import numpy as np


class StepwiseRegress(RidgeRegress):

    def train(self, epsilon, iterations):
        if self._s_calc is False:
            self._calc_standard_matrix()
            self._s_calc = True
        m, n = np.shape(self._s_x_mat)
        weight_arrays = []
        step_weight_mat = np.zeros((n, 1))
        for i in range(iterations):
            # get the min weight matrix of every step
            min_error = float("inf")
            min_weight_mat = None
            for j in range(n):
                for sign in [-1, 1]:
                    tmp_weight_mat = step_weight_mat.copy()  # use weight of previous step
                    tmp_weight_mat[j] += epsilon * sign
                    tmp_y_mat = self._s_x_mat * tmp_weight_mat
                    error = np.square(self._s_y_mat - tmp_y_mat).sum()
                    if error < min_error:
                        min_error = error
                        min_weight_mat = tmp_weight_mat
            # save current weight
            step_weight_mat = min_weight_mat
            # add min_weight_mat to weight_arrays
            weight_arrays.append(step_weight_mat.copy().tolist())
        return weight_arrays


