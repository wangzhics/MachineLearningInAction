__author__ = 'WangZhi'

import os

from test.kNN.digitsfile import *
from kNN.neighbor import Neighbor
from kNN.euclidean import EuclideanAlgorithmFactory


def get_neighbors(file_folder_path):
    neighbors = list()
    for file_path in get_sub_file(file_folder_path):
        label = get_file_label(file_path)
        path = os.path.join(file_folder_path, file_path)
        properties = get_properties(path)
        neighbor = Neighbor(label, properties)
        neighbors.append(neighbor)
    return neighbors

if __name__ == '__main__':
    neighbors = get_neighbors("digits/trainingDigits")
    for file_path in get_sub_file("digits/testDigits"):
        label = get_file_label(file_path)
        path = os.path.join("digits/testDigits", file_path)
        target_properties = get_properties(path)
        euclidean_factory = EuclideanAlgorithmFactory()
        euclidean_algorithm = euclidean_factory.get_algorithm()
        print(file_path + " should be " + label +
              ", the result is " + euclidean_algorithm.get_nearest(target_properties, neighbors, 5))
