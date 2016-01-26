import numpy as np
from pca.core import PCA


def load_data(file_name):
    data_list = []
    fr = open(file_name)
    for line in fr.readlines():
        line_strings = line.strip().split("\t")
        line_floats = list(map(lambda x: float(x), line_strings))
        data_list.append(line_floats)
    return data_list

if __name__ == '__main__':
    data = load_data("testSet.txt")
    data_mat = np.mat(data)
    pca = PCA(data_mat, 1)
    print(pca.get_low_dimension())
    print(pca.new_coordinate())