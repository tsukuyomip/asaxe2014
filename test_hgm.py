#coding: utf-8

import unittest
import numpy as np
from hierarchical_model import BinTree
from test_double.randdouble import RandDouble

class TestBinTree(unittest.TestCase):
    def test_can_born_two_children(self):
        r = RandDouble()

        bt = BinTree(rng = r)
        self.assertEqual(
            (-1, 1),
            bt.born_two_children(1)
        )

    def test_can_born_list_children(self):
        r = RandDouble()

        bt = BinTree(rng = r)
        self.assertEqual(
            bt.born_next_children([-1, 1, 1, -1]),
            [-1,1 , 1,-1 , 1,-1 , -1,1]
        )

    def test_can_make_a_bin_tree_with_inputed_depth(self):
        r = RandDouble()

        bt = BinTree(rng = r, depth = 4)
        self.assertEqual(
            bt.make_a_tree(1),
            [1,-1, -1,1, -1,1, 1,-1]
        )

    def test_can_make_all_bin_tree_with_inputed_depth_and_prob(self):
        r = RandDouble()

        bt = BinTree(rng = r, parents = [1, -1, -1], depth = 4, p = 0.99)
        self.assertEqual(
            bt.make_all_tree(),
            [[1,-1, -1,1, -1,1, 1,-1],
             [-1,1, 1,-1, 1,-1, -1,1],
             [-1,1, 1,-1, 1,-1, -1,1]]
        )

    def test_return_formatted_output_dataset(self):
        r = RandDouble()

        bt = BinTree(rng = r, parents = [1, -1, -1], depth = 4, p = 0.99)
        self.assertTrue(
            np.array_equal(
                bt.make_dataset(),
                np.array([
                    [1, -1, -1],
                    [-1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, 1]])
            )
        )

if __name__ == "__main__":
    unittest.main()
