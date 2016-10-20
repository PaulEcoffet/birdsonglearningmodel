import unittest
import numpy as np
from hill_climbing import hill_climbing

class TestHillClimbing(unittest.TestCase):

    def test_linear(self):
        f = (lambda x: 2*x)
        x, y = hill_climbing(f, np.array([3]), np.array([0]), goal_delta=0.0001, seed=42)
        np.testing.assert_array_almost_equal(x, np.array([1.5]), decimal=4)
        np.testing.assert_array_almost_equal(y, np.array([3]), decimal=4)
        x, y = hill_climbing(f, np.array([3]), np.array([4]), goal_delta=0.0001, seed=42)
        np.testing.assert_array_almost_equal(x, np.array([1.5]), decimal=4)
        np.testing.assert_array_almost_equal(y, np.array([3]), decimal=4)

    def test_can_get_stuck(self):
        f = (lambda x: x**3 - x)
        x, y = hill_climbing(f, np.array([6]), np.array([-1]), goal_delta=0.0001, seed=42)
        self.assertNotAlmostEqual(x[0], 2)
        self.assertNotAlmostEqual(y[0], 4)
        x, y = hill_climbing(f, np.array([6]), np.array([4]), goal_delta=0.0001, seed=42)
        np.testing.assert_array_almost_equal(x, np.array([2]), decimal=4)
        np.testing.assert_array_almost_equal(y, np.array([6]), decimal=4)

    def test_mult_dim(self):
        f = (lambda x: np.array((2 * (x[0] * x[1]), 3 * (-x[1]))))
        x, y = hill_climbing(f, np.array([1, 2]), np.array([-1, 4]), goal_delta=0.0001, seed=42)
        np.testing.assert_array_almost_equal(x, np.array([-3/4, -2/3]), decimal=4)
        np.testing.assert_array_almost_equal(y, np.array([1, 2]), decimal=4)
