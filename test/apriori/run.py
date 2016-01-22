from apriori.core import AprioriSet


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

if __name__ == '__main__':
    data_set = load_data_set()
    apriori = AprioriSet(data_set)
    frequent_list = apriori.get_frequent_set(0.5)
    for frequent in frequent_list:
        print(frequent)
    print("get related list of %s :" % str(frequent_list[-2]))
    related_list = apriori.get_related_set(frequent_list[-2], 0.6)
    for related in related_list:
        print(related)
    print("get related list of %s :" % str(frequent_list[-1]))
    related_list = apriori.get_related_set(frequent_list[-1], 0.6)
    for related in related_list:
        print(related)
