__author__ = 'WangZhi'

from decision.core import DecisionSet
from decision.tree import build_tree
from test.decision.lenses.reader import read_decision_paths

if __name__ == '__main__':
    decision_paths = read_decision_paths()
    decision_set = DecisionSet(decision_paths)
    tree = build_tree(decision_set)
    print(tree)
