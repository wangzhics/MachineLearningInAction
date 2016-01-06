from decision.tree import *
from test.decision.lenses.reader import *


if __name__ == '__main__':
    data_frame = read_data_frame()
    tree = build_tree(data_frame)
    print(tree)