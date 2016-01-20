import numpy as np
import matplotlib.pyplot as plt
from kmeans.core import SimpleKMeans


def load_data(file_name):
    data_list = []
    fr = open(file_name)
    for line in fr.readlines():
        line_strings = line.strip().split("\t")
        line_floats = list(map(lambda x: float(x), line_strings))
        data_list.append(line_floats)
    return data_list

def _get_axis_list(cluster):
    x_list = []
    y_list = []
    for e in cluster.elements:
        e_list = e.tolist()
        x_list.append(e_list[0][0])
        y_list.append(e_list[0][1])
    return x_list, y_list

if __name__ == '__main__':
    data_array = load_data("testSet.txt")
    data_mat = np.mat(data_array)
    simple = SimpleKMeans(data_mat)
    clusters = simple.cluster(4)
    # paint
    filled_markers = ['o', 's', 'd', '^']
    for i in range(4):
        cluster = clusters[i]
        centroid = cluster.centroid.tolist()
        x_list, y_list = _get_axis_list(cluster)
        plt.scatter(x_list, y_list, c="green", s=15, marker=filled_markers[i])
        plt.scatter(centroid[0][0], centroid[0][1], c="black", s=100, marker="+")
    plt.savefig("simple.png")
