import numpy as np
from pca.core import PCA
import matplotlib.pyplot as plt


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
    new_coordinate = pca.new_coordinate()
    new_coordinate_mat = np.mat(new_coordinate)
    old_x = data_mat[:, 0].tolist()
    old_y = data_mat[:, 1].tolist()
    new_x = new_coordinate_mat[:, 0].tolist()
    new_y = new_coordinate_mat[:, 1].tolist()
    plt.scatter(old_x, old_y, c="green", s=10, marker="^", alpha=0.8)
    plt.scatter(new_x, new_y, c="blue", s=10, marker="o")
    plt.savefig("pca.png")
