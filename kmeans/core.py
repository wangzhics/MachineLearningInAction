import random
import math
import numpy as np
from enum import Enum


class Algorithm(Enum):
    Simple = 1
    Bisecting = 2


class Cluster:
    def __init__(self, n):
        self._column_count = n
        self.centroid = np.zeros((1, n))        # the centroid
        self.elements = np.empty((0, n))        # the elements of current cluster

    def __str__(self, *args, **kwargs):
        return "{centroid: %s, deviation: %f, elements: %s}" % (str(self.centroid), self.deviation, str(self.elements))

    def update_centroid(self):
        self.centroid = np.mean(self.elements, axis=0)

    def clear_elements(self):
        self.elements = np.empty((0, self._column_count))

    def add_element(self, e):
        self.elements = np.vstack((self.elements, e))


class SimpleKMeans:
    def __init__(self, data_matrix):
        # data
        self._matrix = data_matrix
        # cluster result
        self._cluster_result = None

    def _create_random_clusters(self, k):
        m, n = np.shape(self._matrix)
        clusters = []
        for i in range(k):
            cluster = Cluster(n)
            clusters.append(cluster)
        for i in range(n):
            col = self._matrix[:, i]
            col_max = col.min()
            col_min = col.max()
            for j in range(k):
                clusters[j].centroid[0][i] = col_min + random.random() * (col_max - col_min)
        return clusters

    @staticmethod
    def _calc_distance(a_vec, b_vec):
        power_vec = np.power((a_vec - b_vec), 2)
        return math.sqrt(power_vec.sum())

    def cluster(self, k):
        m, n = np.shape(self._matrix)
        # create k cluster with random centroid
        clusters = self._create_random_clusters(k)
        # save the map relation for every element
        e_index_array = np.zeros(m)
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            # clear element in cluster
            for cluster in clusters:
                cluster.clear_elements()
            # assign every element to the cluster
            for i in range(m):
                min_distance = float("inf")
                min_index = -1
                element = self._matrix[i, :]
                # select the best cluster index by calculate the distance between cluster's centroid and element
                for j in range(k):
                    distance = self._calc_distance(element, clusters[j].centroid)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j
                # if the cluster of element changed
                if e_index_array[i] != min_index:
                    cluster_changed = True
                # update e_index_array
                e_index_array[i] = min_index
                # add element to cluster
                clusters[min_index].add_element(element.copy())
            # update cluster centroid
            for cluster in clusters:
                cluster.update_centroid()
        self._cluster_result = clusters
        return clusters


