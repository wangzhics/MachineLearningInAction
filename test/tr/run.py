from tr.core import Algorithm
from tr.tree import build_tree
import numpy as np


def load_data(file_name):
    data_list = []
    fr = open(file_name)
    for line in fr.readlines():
        line_strings = line.strip().split("\t")
        line_floats = list(map(lambda x: float(x), line_strings))
        data_list.append(line_floats)
    return data_list

if __name__ == '__main__':
    """
    data_array_1 = load_data("ex0.txt")
    data_mat_1 = np.mat(data_array_1)
    tree_1 = build_tree(data_mat_1, algorithm=Algorithm.RegressTree)
    print(tree_1)
    """
    data_array_2 = load_data("exp2.txt")
    data_mat_2 = np.mat(data_array_2)
    tree_2 = build_tree(data_mat_2, algorithm=Algorithm.ModelTree, min_err=1, min_row_count=10)
    print(tree_2)
