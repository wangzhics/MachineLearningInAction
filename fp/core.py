class HeadNode:
    def __init__(self, item):
        self.item = item
        self.item_set = frozenset([item])
        self.count = 0
        self.next = None


class TreeNode:
    def __init__(self, item):
        self.item = item
        self.item_set = frozenset([item])
        self.count = 1
        self.next = None
        self.parent = None
        self.children = []

    def get_child(self, c_item):
        for child in self.children:
            if child.item == c_item:
                return child
        return None

    def add_child(self, child):
        self.children.append(child)


class FPTree:
    def __init__(self, data_lists, min_count):
        self._data_lists = data_lists
        self._build_data_set_list()
        self._build_head_list(min_count)
        self._build_f_data_lists()
        self._build_tree()

    def get_head(self):
        return self._head_list

    def get_tree(self):
        return self._root_node

    def _build_data_set_list(self):
        self._data_set_list = []
        for data_list in self._data_lists:
            self._data_set_list.append(frozenset(data_list))

    def _build_head_list(self, min_count):
        self._head_list = []
        low_set = set()
        individual_set = set()
        for data_list in self._data_lists:
            for data in data_list:
                individual_set.add(data)
        for individual in individual_set:
            head_node = HeadNode(individual)
            head_node.count = self._calc_count(head_node.item_set)
            if head_node.count >= min_count:
                self._head_list.append(head_node)
            else:
                low_set.add(individual)
        self._head_list.sort(key=lambda x: x.count, reverse=True)
        # item to remove
        self._low_set = frozenset(low_set)

    def _build_f_data_lists(self):
        self._f_data_lists = []
        for data_set in self._data_set_list:
            tmp_set = data_set - self._low_set
            tmp_list = []
            for head in self._head_list:
                if head.item_set.issubset(tmp_set):
                    tmp_list.append(head.item)
            self._f_data_lists.append(tmp_list)

    def _build_tree(self):
        self._root_node = TreeNode('root')
        for f_data_list in self._f_data_lists:
            self._build_f_tree(self._root_node, f_data_list)

    def _build_f_tree(self, p_node, data_list):
        for data in data_list:
            tree_node = TreeNode(data)
            child = p_node.get_child(data)
            # link to the tree
            if child is None:
                # the leaf
                tree_node.parent = p_node
                p_node.add_child(tree_node)
                p_node = tree_node
                # link to the head
                head = self._get_head_node(data)
                while head.next is not None:
                    head = head.next
                head.next = tree_node
            else:
                # the breach
                child.count += 1
                p_node = child

    def _get_head_node(self, item):
        for head in self._head_list:
            if head.item == item:
                return head
        return None

    def _calc_count(self, s):
        n = 0
        for data_set in self._data_set_list:
            if s.issubset(data_set):
                n += 1
        return n