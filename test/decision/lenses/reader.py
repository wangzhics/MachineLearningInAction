__author__ = 'WangZhi'

import os
from decision.core import DecisionPath, DecisionSet

titles = ['age', 'spectacle prescription', 'astigmatic', 'tear rate']


def read_decision_paths():
    decision_paths = []
    try:
        with open("lenses.data", 'r') as file:
            for each_line in file:
                decision_path_list = _build_path(each_line)
                properties = {}
                for index, item in enumerate(decision_path_list):
                    if index == 0 or index == 5:
                        continue
                    properties[titles[index - 1]] = item
                decision_path = DecisionPath(decision_path_list[0], properties, decision_path_list[5])
                decision_paths.append(decision_path)
            return decision_paths
    except IOError as io_error:
        print('Open File[" + file_path + "] Error:' + str(io_error))


def _build_path(path_str):
    paths = []
    tmp = ''
    for s in path_str:
        if s == ' ' or s == '\t' or s == '\n' or s is os.linesep:
             if tmp.strip():
                paths.append(tmp)
                tmp = ''
        else:
            tmp = tmp + s
    return paths
