__author__ = 'WangZhi'

from pandas import Series
from kNN.neighbor import Neighbor, NeighborDistance


class Algorithm():
    """
    k临近算法
    """
    def get_distance(self, target_properties, data_properties):
        """
        获取目标和邻居的距离
        :param target_properties: 查找的目标的属性
        :param data_properties: 遍历的邻居的属性
        :return: 它们之间的距离
        """
        pass

    def get_nearest(self, target_properties, neighbors, top_k):
        """
        获取最近的标签
        :param target_properties: 查找的目标的属性
        :param neighbors: 所有的邻居
        :param top_k: 筛选个数
        :return: 最近的标签
        """
        neighbor_distances = []
        for neighbor in neighbors:
            if not isinstance(neighbor, Neighbor):
                continue
            properties = neighbor.get_properties()
            distance = self.get_distance(target_properties, properties)
            if distance:
                neighbor_distance = NeighborDistance(neighbor, distance)
                neighbor_distances.append(neighbor_distance)
                print(neighbor_distance)
        sorted_neighbor_distances = sorted(neighbor_distances, key=lambda n: n.get_distance())
        labels = []
        for neighbor_distance in sorted_neighbor_distances[0:top_k]:
            labels.append(neighbor_distance.get_neighbor().get_label())
        label_counts = Series(labels).value_counts()
        return label_counts.index[0]


class AlgorithmFactory():
    """
    k临近算法工厂
    """
    def get_algorithm(self):
        """
        获取k临近算法
        :return: k临近算法
        """
        pass

    @staticmethod
    def get_type(self):
        """
        获取算法类型
        :return: 算法类型
        """
        pass

