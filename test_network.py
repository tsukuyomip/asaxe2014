#coding: utf-8

import unittest
import numpy as np

from network import LinearNetwork
from test_double.randdouble import RandDouble

from exception.inputexception import *

class TestBinTree(unittest.TestCase):
    def test_can_memorize_number_of_layers(self):
        network = LinearNetwork(l = 3, n = (10, 3, 5))
        self.assertEqual(network.l, 3)

    def test_can_memorize_number_of_neurons(self):
        network = LinearNetwork(l = 4, n = (9, 7, 2, 3))
        self.assertEqual(network.n, (9, 7, 2, 3))

    def test_create_right_size_of_weight_array(self):
        network = LinearNetwork(l  = 4, n = (10, 5, 6, 10))
        self.assertEqual(network.W[0].shape, (5, 10))
        self.assertEqual(network.W[1].shape, (6, 5))
        self.assertEqual(network.W[2].shape, (10, 6))

    def test_decide_wrong_input_of_n(self):
        try:
            network = LinearNetwork(l  = 4, n = (10, 5, 20))
            self.fail("did not raise")
        except InputException:
            pass

    def test_set_initial_weight_with_upto_inputed_maxw(self):
        network = LinearNetwork(l  = 3, n = (10, 5, 20), maxw = 0.0)
        self.assertEqual(network.W[0].mean(), 0.0)

        network = LinearNetwork(l  = 3, n = (10, 5, 20), maxw = 0.5)
        self.assertNotEqual(network.W[0].mean(), 0.0)

    def test_initialize_weight_with_inputed_W(self):
        W0 = np.array([[3.5, -10, 1.5]])
        W1 = np.array([[ 0.5],
                       [-0.2]])
        network = LinearNetwork(l = 3, n = (3, 1, 2), W =(W0, W1))
        self.assertTrue(
            np.array_equal(network.W[0], W0)
        )
        self.assertTrue(
            np.array_equal(network.W[1], W1)
        )

    def test_calculate_answer_from_input(self):
        input_data = np.array([[3, -6], [2, -4], [1, -2]])
        W0 = np.array([[2, -1, 0.5]])
        W1 = np.array([[ 10],
                       [-10]])
        network = LinearNetwork(l = 3, n = (3, 1, 2), W =(W0, W1))

        ans = network.run(input_data)
        self.assertTrue(
            np.array_equal(np.array([[45.0, -90.0], [-45.0, 90.0]]), ans)
        )

    def test_update_self_weight(self):
        W0 = np.array([[2, -1, 0.5]])
        W1 = np.array([[ 10],
                       [-10]])
        network = LinearNetwork(l = 3, n = (3, 1, 2), W =(W0, W1))

        new_W0 = np.array([[1, 2, 3]])
        network.set_W((new_W0, W1))
        self.assertTrue(
            np.array_equal(network.W[0], new_W0)
        )
        self.assertTrue(
            np.array_equal(network.W[1], W1)
        )

if __name__ == "__main__":
    unittest.main()
