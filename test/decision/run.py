__author__ = 'WangZhi'

from pandas import Series, DataFrame
from decision.tree import *

if __name__ == '__main__':
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    columns = ['no surfacing', 'flipper', 'value']
    df = DataFrame(dataSet, columns=columns)
    entropy_frame = calc_shannon_entropy(df)
    print(entropy_frame)