import random
import math
import numpy as np
from enum import Enum


class Cluster:
    def __init__(self, n):
        self._column_count = n
        self.centroid = np.zeros((1, n))        # the centroid
        self.elements = np.empty((0, n))        # the elements of current cluster
        self.err = 0.0

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
    def calc_distance(a_vec, b_vec):
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
            # clear element and err in cluster
            for cluster in clusters:
                cluster.clear_elements()
                cluster.err = 0.0
            # assign every element to the cluster
            for i in range(m):
                min_distance = float("inf")
                min_index = -1
                element = self._matrix[i, :]
                # select the best cluster index by calculate the distance between cluster's centroid and element
                for j in range(k):
                    distance = self.calc_distance(element, clusters[j].centroid)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j
                # if the cluster of element changed
                if e_index_array[i] != min_index:
                    cluster_changed = True
                # update e_index_array
                e_index_array[i] = min_index
                # add element and err to cluster
                clusters[min_index].add_element(element.copy())
                clusters[min_index].err += min_distance ** 2
            # update cluster centroid
            for cluster in clusters:
                cluster.update_centroid()
        self._cluster_result = clusters
        return clusters


class BisectKMeans:
    def __init__(self, data_matrix):
        # data
        self._matrix = data_matrix
        # cluster result
        self._cluster_result = None

    def cluster(self, k):
        m, n= np.shape(self._matrix)
        cluster_list = []
        # add the central as first cluster centroid
        first_centroid = np.mean(self._matrix, axis=0)
        first_err = 0.0
        for i in range(m):
            first_err += SimpleKMeans.calc_distance(first_centroid, self._matrix[i, :]) ** 2
        first_cluster = Cluster(n)
        first_cluster.centroid = first_centroid
        first_cluster.elements = self._matrix.copy()
        first_cluster.err = first_err
        cluster_list.append(first_cluster)
        # try split cluster in cluster_list
        while len(cluster_list) < k:
            has_split = False
            for i in range(len(cluster_list)):
                cluster = cluster_list[i]
                sub_simple = SimpleKMeans(cluster.elements)
                sub_cluster_list = sub_simple.cluster(2)
                sub_err = sub_cluster_list[0].err + sub_cluster_list[1].err
                if cluster.err > sub_err :  # good to split
                    cluster_list[i] = sub_cluster_list[0]  # replace current from first
                    cluster_list.append(sub_cluster_list[1]) # add second to list tail
                    has_split = True
            if has_split is False:  # can not split anymore
                break
        self._cluster_result = cluster_list
        return cluster_list
