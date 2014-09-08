__author__ = 'WangZhi'

import os
from kNN.neighbor import Neighbor

__all__ = ('get_sub_file', 'get_properties', 'get_file_label')


def get_sub_file(file_folder_path):
    file_list = []
    files = os.listdir(file_folder_path)
    for f in files:
        path = os.path.join(file_folder_path, f)
        if os.path.isfile(path):
            file_list.append(f)
    return file_list


def get_properties(file_path):
    properties = []
    try:
        with open(file_path, 'r') as file:
            for each_line in file:
                for s in each_line:
                    if s is os.linesep or s is '\n':
                        continue
                    properties.append(int(s))
            return properties
    except IOError as io_error:
        print("Open File[" + file_path + "] Error:" + str(io_error))


def get_file_label(file_name):
    names = file_name.split('_')
    return names[0]


