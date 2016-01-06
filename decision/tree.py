from decision.core import *


class Tree:
    """
    the tree root node
    """
    def __init__(self, feature):
        self._name = "Decision Root"
        self._feature = feature
        self._children = []

    def get_name(self):
        return self._name

    def get_feature(self):
        return self._feature

    def get_children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)


class Node:
    """
    the tree node
    """
    def __init__(self, parent, value, feature):
        self._parent = parent
        self._value = value
        self._feature = feature
        self._children = []

    def get_parent(self):
        return self._parent

    def get_value(self):
        return self._value

    def get_feature(self):
        return self._feature

    def get_children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)


def build_tree(data_frame):
    # get result column
    frame_columns = data_frame.columns
    feature_count = len(frame_columns) - 1
    result_title = frame_columns[feature_count]
    base_entropy = calc_shannon_entropy(data_frame[result_title])
    # if base_entropy is 0 means this is a leaf node
    if base_entropy == 0:
        result = data_frame[result_title][0]
        return Tree(result);
    best_feature = get_best_feature(data_frame)
    root = Tree(best_feature)
    sub_frames = split_by_feature(data_frame, best_feature)
    # build sub tree
    for feature_value, sub_frame in sub_frames.items():
        sub_node = _build_sub_tree(root, feature_value, sub_frame)
        root.add_child(sub_node)
    return root


def _build_sub_tree(parent, feature_value, data_frame):
    # get result column
    frame_columns = data_frame.columns
    feature_count = len(frame_columns) - 1
    result_title = frame_columns[feature_count]
    base_entropy = calc_shannon_entropy(data_frame[result_title])
    # if base_entropy is 0 means this is a leaf node
    if base_entropy == 0:
        result = data_frame[result_title][0]
        return Node(parent, feature_value, result)
    best_feature = get_best_feature(data_frame)
    node = Node(parent, feature_value, best_feature)
    sub_frames = split_by_feature(data_frame, best_feature)
    # build sub tree
    for feature_value, sub_frame in sub_frames.items():
        sub_node = _build_sub_tree(node, feature_value, sub_frame)
        node.add_child(sub_node)
    return node

