__author__ = 'WangZhi'


class Neighbor:
    """
    邻居对象
    """
    def __init__(self, label, properties):
        self._label = label
        self._properties = properties

    def get_label(self):
        return self._label

    def get_properties(self):
        return self._properties

    def __str__(self):
        return "{label: " + self._label + ", properties: " + str(self._properties) + "}"


class NeighborDistance:
    """
    和邻居对象的距离
    """
    def __init__(self, neighbor, distance=-1):
        self._neighbor = neighbor
        self._distance = distance

    def get_neighbor(self):
        return self._neighbor

    def get_distance(self):
        return self._distance

    def __str__(self):
        return "{distance: " + str(self._distance) + ", neighbor: " + str(self._neighbor) + "}"


