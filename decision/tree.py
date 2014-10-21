#coding=utf-8
#当有中文注释时，必须加上文件编码信息，否则在Window下调试会出错
__author__ = 'WangZhi'

from pandas import Series, DataFrame
from math import log2


def calc_shannon_entropy(data_frame, value_column='value'):
    index_length = len(data_frame.index)
    #计算每一列的香农熵
    entropy_all = {}
    for col in data_frame.columns:
        if col is value_column:
            continue
        #计算每一列中各个值出现的个数
        col_df = data_frame[col].fillna('None')
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


def build_tree(data_frame):
    return 0

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
