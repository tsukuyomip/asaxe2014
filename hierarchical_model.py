#coding: utf-8
import numpy as np

class BinTree(object):
    def __init__(self, rand = np.random.random, depth = 2, p = 0.99, parents = [1, -1]):
        self.rand = rand
        self.depth = depth
        self.p = p
        self.parents = parents
        pass

    def born_two_children(self, parent):
        l = parent
        if self.rand() > 0.9:
            l = -l
        r = parent
        if self.rand() > 0.9:
            r = -r
        return (l, r)

    def born_next_children(self, parent_list):
        ret_list = []
        for item in parent_list:
            for i in xrange(2):
                if self.rand() > 0.9:
                    ret_list.append(item)
                else:
                    ret_list.append(-item)

        return ret_list

    def make_a_tree(self, parent):
        ret_list = []
        ret_list.append(parent)

        for i in xrange(self.depth - 1):  # ルートノードのみなら depth=1なので
            ret_list = self.born_next_children(ret_list)

        return ret_list

    def make_all_tree(self):
        ret_list = []
        for p in self.parents:
            ret_list.append(self.make_a_tree(p))
        return ret_list

    def make_dataset(self):
        return np.array(self.make_all_tree()).T
