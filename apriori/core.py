import copy


class RelatedSet:
    def __init__(self, p_set, h_set, degree):
        self.p_set = p_set
        self.h_set = h_set
        self.degree = degree

    def __str__(self, *args, **kwargs):
        return "{'p_set': %s, 'p_set': %s, 'degree:' %f}" % (str(self.p_set), str(self.h_set), self.degree)



class SupportSet:
    def __init__(self, a_set, support):
        self.a_set = a_set
        self.support = support

    def __str__(self, *args, **kwargs):
        return "{'a_set': %s, 'support:' %f}" % (str(self.a_set), self.support)


class AprioriSet:
    def __init__(self, data_lists):
        self._data_lists = data_lists
        # create data_set_list and individual_set_list
        self._m = len(self._data_lists)
        self._data_set_list = []
        self._individual_set_list = []
        individual_set = set()                  # temp variables
        for data_list in data_lists:
            for data in data_list:
                individual_set.add(data)
            self._data_set_list.append(frozenset(data_list))
        individual_list = list(individual_set)  # temp variables
        individual_list.sort()
        for individual in individual_list:
            s = frozenset([individual])
            f = self.calc_support(s)
            self._individual_set_list.append(SupportSet(s, f))

    def calc_support(self, s):
        n = 0.0
        for data_set in self._data_set_list:
            if s.issubset(data_set):
                n += 1
        support = n / self._m
        return support

    def _build_join_sets(self, priori_list):
        k = len(priori_list)
        if k < 2:
            return []
        join_set_list = []
        join_set = set()
        for i in range(k - 1):
            for j in range(i + 1, k):
                tmp_set = priori_list[i].a_set | priori_list[j].a_set
                tmp_set = frozenset(tmp_set)
                if tmp_set in join_set:
                    continue
                tmp_set = frozenset(tmp_set)
                join_set_list.append(SupportSet(tmp_set, self.calc_support(tmp_set)))
                join_set.add(tmp_set)
        return join_set_list

    def get_frequent_set(self, min_f=0.5):
        frequent_list = []
        priori_list = self._individual_set_list
        for i in range(self._m):
            if len(priori_list) == 0:
                break
            tmp_list = []
            # add priori_list
            for priori in priori_list:
                if priori.support >= min_f:
                    if len(priori.a_set) == 1:
                        tmp_list.append(copy.copy(priori))
                    else:
                        tmp_list.append(priori)
            if len(tmp_list) == 0:
                break
            frequent_list.extend(tmp_list)
            # build next level set
            priori_list = self._build_join_sets(tmp_list)
        return frequent_list

    def _build_split_sets(self, full_set_list, min_d):
        split_set_list = set()
        split_set = set()
        for full_set in full_set_list:
            p_len = len(full_set.p_set)
            if p_len < 2:
                continue
            p_list = list(full_set.p_set)
            for p in p_list:
                # temporary set
                t_p_set = set(full_set.p_set.copy())
                t_h_set = set(full_set.h_set.copy())
                t_p_set.remove(p)
                t_h_set.add(p)
                # if f_p_set already exists
                f_p_set = frozenset(t_p_set)
                if f_p_set in split_set:
                    continue
                split_set.add(f_p_set)
                # build RelatedSet
                f_h_set = frozenset(t_h_set)
                p_support = self.calc_support(f_p_set)
                full_support = self.calc_support((f_p_set | f_h_set))
                degree =  full_support / p_support
                if degree >= min_d:
                    r_set = RelatedSet(f_p_set, f_h_set, degree)
                    split_set_list.add(r_set)
        return split_set_list

    def get_related_set(self, support_set, min_d=0.5):
        related_set_list = []
        root_set = RelatedSet(support_set.a_set, frozenset(), 0)
        itr_list = [root_set]
        for i in range(len(support_set.a_set)):
            tmp_list = self._build_split_sets(itr_list,min_d)
            related_set_list.extend(tmp_list)
            itr_list = tmp_list
        return related_set_list







