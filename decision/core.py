#coding=utf-8
#当有中文注释时，必须加上文件编码信息，否则在Window下调试会出错
__author__ = 'WangZhi'

import copy
from pandas import Series, DataFrame
from math import log2


class DecisionPath:
    """
    决策项，该对象包括索引，决策属性，决策结果三个属性
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

    def equal_properties(self, other):
        if other is None:
            return False
        other_properties = other.get_properties()
        if len(self._properties) != len(other_properties):
            return False
        for (k, v) in self._properties.items():
            if v != other_properties[k]:
                return False
        return True

    def __str__(self):
        return '{index:' + self._index + ', properties:' + str(self._properties) + \
               ', decision:' + self._decision + '}'

    def __eq__(self, other):
        if not isinstance(other, DecisionPath):
            return False
        if not (self._index and self._decision):
            return False
        if self._index != other._index:
            return False
        if self._decision != other._decision:
            return False
        other_properties = other.get_properties()
        if len(self._properties) != len(other_properties):
            return False
        for i in len(self._properties):
            if self._properties[i] != other_properties[i]:
                return False
        return True


class DecisionSet:
    """
    决策集合
    """
    def __init__(self, decision_paths):
        self._decision_paths = []
        if decision_paths:
            self.add_paths(decision_paths)

    def add_paths(self, decision_paths):
        for decision_path in decision_paths:
            self.add_path(decision_path)

    def add_path(self, decision_path):
        if len(self._decision_paths) > 0:
            # 检查是否重复
            for self_path in self._decision_paths:
                if self_path.equal_properties(decision_path):
                    if self_path.get_decision() == decision_path.get_decision():
                        #不添加重复的决策路径，相同的的决策属性，相同的决策结果
                        print('决策路径【' + str(decision_path.get_index) +
                                '】和【' + self_path.get_index() + '】有相同的的决策属性和决策结果，不重复添加')
                    else:
                        #相同的的决策属性，不同的决策结果抛出异常
                        raise DecisionException('决策路径【' + str(decision_path.get_index) +
                                '】和【' + self_path.get_index() + '】有相同的的决策属性却有不同的决策结果')
            #检查决策属性个数是否一致
            if len(self._decision_paths[0].get_properties()) != len(decision_path.get_properties()):
                raise DecisionException('决策路径【' + str(decision_path.get_index) +
                        '】决策属性个数不一致，默认为【' + str(len(self._decision_paths[0].get_properties())) + '】，但其为【' + str(len(decision_path.get_properties())) + '】')
        self._decision_paths.append(decision_path)

    def get_decision_paths(self):
        return self._decision_paths

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
        sub_decision_paths = {}
        for decision_path in self._decision_paths:
            new_decision_path = copy.deepcopy(decision_path)
            pro_value = new_decision_path.pop_property(property_name)
            if not pro_value in sub_decision_paths:
                sub_decision_paths[pro_value] = []
            sub_decision_paths[pro_value].append(new_decision_path)
        for (value, sub_paths) in sub_decision_paths.items():
            sub_sets[value] = DecisionSet(sub_paths)
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


class DecisionException(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self._msg = msg