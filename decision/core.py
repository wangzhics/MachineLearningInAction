#coding=utf-8
#当有中文注释时，必须加上文件编码信息，否则在Window下调试会出错
__author__ = 'WangZhi'

import copy
from pandas import Series, DataFrame
from math import log2


class DecisionPath:
    """
    决策项，该对象包括索引（请确保唯一），决策属性，决策结果三个属性
    """
    def __init__(self, index, properties, decision):
        self._index = index
        self._properties = properties
        self._decision = decision

    def get_index(self):
        return self._index

    def get_properties(self):
        return self._properties

    def get_decision(self):
        return self._decision

    def pop_property(self, property_name):
        return self._properties.pop(property_name)

    def __str__(self):
        return '{index:' + self._index + ', properties:' + str(self._properties) + \
               ', decision:' + self._decision + '}'


class DecisionSet:
    """
    决策集合
    """
    def __init__(self, decision_paths=[]):
        self._decision_paths = []
        self._decision_paths_dict = {}
        if decision_paths:
            for decision_path in decision_paths:
                self.add_path(decision_path)

    def add_path(self, decision_path):
        self._decision_paths.append(decision_path)
        self._decision_paths_dict = decision_path

    def get_by_index(self, index):
        return self._decision_paths_dict.get(index)

    def find_best_property(self):
        """
        根据每个决策属性的香农熵，选择最适合分割的决策属性
        :return:最适合分割的决策属性
        """
        decision_index = []
        decision_series = []
        for decision_path in self._decision_paths:
            decision_index.append(decision_path.get_index())
            decision_series.append(Series(decision_path.get_properties()))
        decision_frame = DataFrame(decision_series, index=decision_index)
        entropy_frame = self._calc_shannon_entropy(decision_frame)
        return entropy_frame.index[0]

    def split_by_property(self, property_name):
        """
        根据选择的决策属性分割决策集合成为几个子集合
        :param property_name:选择的决策属性名称
        :return:决策子集合
        """
        sub_sets = {}
        for decision_path in self._decision_paths:
            new_decision_path = copy.deepcopy(decision_path)
            pro_value = new_decision_path.pop_property(property_name)
            if not pro_value in sub_sets:
                sub_sets[pro_value] = []
            sub_sets[pro_value].append(new_decision_path)
        return sub_sets

    def _calc_shannon_entropy(self, decision_frame):
        index_length = len(decision_frame.index)
        #计算每一列的香农熵
        entropy_all = {}
        for col in decision_frame.columns:
            #计算每一列中各个值出现的个数
            col_df = decision_frame[col].fillna('None')
            count_series = col_df.value_counts()
            #计算熵值
            shannon_entropy = 0.0
            for c in count_series:
                prob = float(c)/index_length
                shannon_entropy -= prob * log2(prob)
            entropy_all[col] = shannon_entropy
        entropy_frame = DataFrame(Series(entropy_all), columns=['entropy'])
        #按降序排序
        return entropy_frame.sort(columns='entropy', ascending=False)