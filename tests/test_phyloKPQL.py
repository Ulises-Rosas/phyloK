import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phyloKPQL import KRGEE


class TestKRGEE(unittest.TestCase):
    def test_get_r_rejects_non_positive_diagonal(self):
        model = KRGEE(get_corr=True)
        vcv = np.array([[0.0, 1.0], [1.0, 2.0]])

        with self.assertRaises(ValueError):
            model.get_R(vcv)

    def test_get_r_returns_correlation_matrix_for_valid_input(self):
        model = KRGEE(get_corr=True)
        vcv = np.array([[1.0, 0.5], [0.5, 1.0]])

        result = model.get_R(vcv)

        np.testing.assert_allclose(np.diag(result), np.ones(2))
        np.testing.assert_allclose(result[0, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
