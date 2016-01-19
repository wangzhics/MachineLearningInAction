from tr.core import *
import numpy as np


class TreeNode:
    def __init__(self):
        # attribute of leaf node
        self.value = None
        # attribute of branch node
        self.split_col_index = None
        self.split_col_value = None
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        b_left = (self.left_child is None)
        b_right = (self.right_child is None)
        b = (b_left and b_right)
        return b

    def __str__(self, *args, **kwargs):
        if self.is_leaf():
            return "{value: %s}" % str(self.value)
        else:
            return "{col_index: %d, col_value: %f, left_child: %s, right_child %s}" \
                   % (self.split_col_index, self.split_col_value, str(self.left_child), str(self.right_child))


def build_tree(parent_mat, algorithm=Algorithm.RegressTree, min_err=1, min_row_count=4):
    parent_node = TreeNode()
    best_index, best_value = get_best_split(parent_mat, min_err, min_row_count)
    if best_index == -1:
        parent_node.value = best_value
        return parent_node
    # split the parent matrix
    left_mat, right_mat = split_by_col(parent_mat, best_index, best_value)
    left_node = build_tree(left_mat, algorithm, min_err, min_row_count)
    right_node = build_tree(right_mat, algorithm, min_err, min_row_count)
    parent_node.split_col_index = best_index
    parent_node.split_col_value = best_value
    parent_node.left_child = left_node
    parent_node.right_child = right_node
    return parent_node





