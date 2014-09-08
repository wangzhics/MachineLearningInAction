__author__ = 'WangZhi'
"""
使用欧式距离作为近邻的距离度量表示法
"""

import math
from kNN.algorithm import Algorithm, AlgorithmFactory


class EuclideanAlgorithm(Algorithm):
    """
    以欧氏距离（ Euclidean distance）作为临近距离
    """
    def get_distance(self, target_properties, data_properties):
        i = 0
        distance = 0
        for num in target_properties:
            distance += math.pow((num - data_properties[i]), 2)
            i += 1
        distance = math.sqrt(distance)
        return distance


class EuclideanAlgorithmFactory(AlgorithmFactory):
    TYPE = "Euclidean_Distance"

    def get_algorithm(self):
        return EuclideanAlgorithm()

    def get_type(self):
        return self.TYPE





