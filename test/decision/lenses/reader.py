__author__ = 'WangZhi'

import os
import pandas as pd

titles = ['index', 'age', 'spectacle prescription', 'astigmatic', 'tear rate', 'result']


def read_data_frame():
    data_rows = []
    try:
        with open('lenses/lenses.data', 'r') as file:
            data_row = []
            tmp_str = ''
            for each_line in file:
                for s in each_line:
                    if s is ' ' or s is os.linesep or s is '\n':
                        if len(tmp_str) > 0:
                            data_row.append(int(tmp_str))
                            tmp_str = ''
                    else:
                        tmp_str += s
                    if s is os.linesep or s is '\n':
                        data_rows.append(data_row)
                        data_row = []

        data_frame = pd.DataFrame(data_rows, columns=titles)
        data_frame.pop('index')
        return data_frame
    except IOError as io_error:
        print(str(io_error))
