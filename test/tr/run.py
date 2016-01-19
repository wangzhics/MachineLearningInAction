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
    data_array = load_data("ex0.txt")
    data_mat = np.mat(data_array)
    tree = build_tree(data_mat, algorithm=Algorithm.RegressTree)
    print(tree)
