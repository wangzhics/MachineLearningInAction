__author__ = 'WangZhi'

from decision.core import DecisionSet


class LeafNode():
    def __init__(self, decision):
        self._decision = decision

    def get_decision(self):
        return self._decision


class BranchNode():
    def __init__(self, property):
        self._property = property
        self._children = {}

    def add_child(self, value, node):
        self._children[value] = node


def build_tree(root_set):
    all_paths = root_set.get_decision_paths()
    #如果没有决策路径则返回空
    if len(all_paths) == 0:
        return None
    #如果决策集合只有一个决策路径
    if len(all_paths) == 1:
        return LeafNode(all_paths[0].get_decision())
    property_name = root_set.find_best_property();
    root = BranchNode(property_name)
    sub_sets = root_set.split_by_property(property_name)
    for (pro_value, sub_set) in sub_sets.items():
        sub_node = build_sub_tree(sub_set)
        root.add_child(pro_value, sub_node)
    return root

def build_sub_tree(path_set):
    if is_end(path_set):
        return LeafNode(path_set.get_decision_paths()[0].get_decision())
    else:
        property_name = path_set.find_best_property();
        node = BranchNode(property_name)
        sub_sets = path_set.split_by_property(property_name)
        for (pro_value, sub_set) in sub_sets.items():
            sub_node = build_sub_tree(sub_set)
            node.add_child(pro_value, sub_node)
        return node

def is_end(path_set):
    all_paths = path_set.get_decision_paths()
    if len(all_paths) == 1:
        return True
    return False

