import numpy as np
from pandas import DataFrame
from boosting.stump import DecisionStump


class AdaptiveBoost:
    def __init__(self):
        # train data
        self._train_frame = DataFrame({"feature": [], "label": []})
        # train result
        self._best_features = []
        self._alphas = []
        self._error_rate = 1

    def set_train_frame(self, train_frame):
        self._train_frame["feature"] = train_frame["feature"]
        self._train_frame["label"] = train_frame["label"]

    def add_train_features(self, train_features, train_labels):
        for i in range(len(train_features)):
            self._train_frame = \
                self._train_frame.append({"feature": train_features[i], "label": train_labels[i]}, ignore_index=True)

    def train(self, max_iteration):
        feature_mat = np.mat(self._train_frame["feature"].tolist())
        label_mat = np.mat(self._train_frame["label"].tolist()).transpose()
        m, n = np.shape(feature_mat)
        weights = np.ones((m, 1)) / m
        results = np.zeros((m, 1))
        for i in range(max_iteration):
            stump = DecisionStump(feature_mat, label_mat)
            best_feature, best_result, best_error = stump.get_best_classify(weights)
            # calculate the alpha
            alpha = 0.5 * np.log((1 - best_error) / max(best_error, 1e-16))
            self._best_features.append(best_feature)
            self._alphas.append(alpha)
            # update weights
            expon = np.multiply(-1 * alpha * label_mat, best_result)
            weights = np.multiply(weights, np.exp(expon))
            weights = weights / weights.sum()
            # update results
            results += alpha * best_result
            # calculate the error rate
            error_vec = np.multiply(np.sign(results) != label_mat, np.ones((m, 1)))
            error_rate =  error_vec.sum() / m
            self._error_rate = error_rate
            if error_rate == 0.0:
                break

    def classify(self, feature_list):
        feature_mat = np.mat(feature_list)
        m, n = np.shape(feature_mat)
        result = np.zeros((m, 1))
        for i in range(len(self._best_features)):
            best_feature = self._best_features[i]
            best_result = DecisionStump.classify(feature_mat, best_feature.index, best_feature.value)
            result += self._alphas[i] * best_result
        result = np.sign(result)
        return result


