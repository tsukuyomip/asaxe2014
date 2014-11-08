#coding: utf-8
import numpy as np

class BinTree(object):
    def __init__(self, rng = np.random, depth = 4, p = 0.01, parents = [1, -1, -1, 1, 1, 1]):
        """
        depth  : ルートノードを含む二分木の深さ
        p      : 子が反転する確率
        parents: 親のリスト．作る木の数と同じ長さ．
        """
        self.rng = rng
        self.depth = depth
        self.p = p
        self.parents = parents
        pass

    def born_two_children(self, parent):
        l = parent
        if self.rng.rand() > self.p:
            l = -l
        r = parent
        if self.rng.rand() > self.p:
            r = -r
        return (l, r)

    def born_next_children(self, parent_list):
        ret_list = []
        for item in parent_list:
            for i in xrange(2):
                if self.rng.rand() > self.p:
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

    def make_dataset(self):  ## いらない可能性が！！！
        return np.array(self.make_all_tree()).T
