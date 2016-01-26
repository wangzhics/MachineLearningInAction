from fp.core import FPTree


def load_simple_data():
    simple_data = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_data

if __name__ == '__main__':
    data_set = load_simple_data()
    new_data_set = []
    for data in data_set:
        new_data_set.append([data, 1])
    fp = FPTree(new_data_set, 3)
    fp.get_frequency_set(frozenset([]))