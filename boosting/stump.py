import numpy as np
from pandas import DataFrame


class BestFeature:
    def __init__(self, feature_index, feature_value):
        self.index = feature_index
        self.value = feature_value


class DecisionStump:
    def __init__(self, feature_mat, label_mat):
        self._feature_mat = feature_mat
        self._label_mat = label_mat

    @staticmethod
    def classify(feature_mat, feature_index, feature_value):
        feature_count = np.shape(feature_mat)[0]
        result_array = np.ones((feature_count, 1))
        result_array[feature_mat[:, feature_index] <= feature_value] = -1.0
        return result_array

    def get_best_classify(self, weights):
        m, n = np.shape(self._feature_mat)
        # prepare parameters
        step_count = 10
        best_feature = BestFeature(-1, 0)
        best_result = np.zeros((m, 1))
        best_weight_error = 1.0
        # classify by each feature
        for i in range(n):
            feature_row = self._feature_mat[:, i]
            step_size = (feature_row.max() - feature_row.min()) / float(step_count)
            for j in range(-1, step_count + 1):
                classify_value = feature_row.min() + step_size * j
                classify_result = self.classify(self._feature_mat, i, classify_value)
                classify_error = np.mat(np.ones((m, 1)))
                classify_error[classify_result == self._label_mat] = 0
                weight_error = weights.transpose() * classify_error
                weight_error = np.sum(weight_error)
                if weight_error < best_weight_error:
                    best_weight_error = weight_error
                    best_feature = BestFeature(i, classify_value)
                    best_result = classify_result
        return best_feature, best_result, best_weight_error



