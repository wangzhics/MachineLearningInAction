import numpy as np


class PCA:
    def __init__(self, data_mat, dimension_count):
        mean_mat = np.mean(data_mat, axis=0)  # calculate the mean by column
        mean_error_mat = data_mat - mean_mat
        cov_error_mat = np.cov(mean_error_mat, rowvar=0)
        eigen_value, eigen_vector = np.linalg.eig(cov_error_mat)
        # sort by eigen_value
        value_index_list = []
        j = 0
        for value in eigen_value:
            value_index_list.append([value, j])
            j += 1
        value_index_list.sort(key=lambda x: x[0], reverse=True)
        # save top k eigen_value and eigen_vector
        eigen_value_list = []
        eigen_vector_list= []
        for i in range(dimension_count):
            eigen_value_list.append(value_index_list[i][0])
            eigen_vector_list.append(eigen_vector[:, value_index_list[i][1]])
        self._eigen_value_mat = np.mat(eigen_value_list)
        self._eigen_vector_mat = np.mat(eigen_vector_list).transpose()
        # lower dimension
        self._lower_mat = mean_error_mat * self._eigen_vector_mat
        # coordinate transfrom
        self._coordinate_mat = self._lower_mat * self._eigen_vector_mat.transpose() + mean_mat

    def get_low_dimension(self):
        return self._lower_mat.tolist()

    def new_coordinate(self):
        return self._coordinate_mat.tolist()


