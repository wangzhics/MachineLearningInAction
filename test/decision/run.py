__author__ = 'WangZhi'

import os
from pandas import Series, DataFrame
from decision.tree import *



if __name__ == '__main__':
    # decision_paths = read_decision_paths("data\\lenses.txt")
    columns = ['no surfacing', 'flipper', 'value']
    # df = DataFrame(decision_paths, columns=columns)
    # entropy_frame = calc_shannon_entropy(df)
    # print(entropy_frame)