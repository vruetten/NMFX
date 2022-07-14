import unittest
import numpy as np
from nmfx import nmf
from nmfx.parameters import Parameters
from nmfx.utils import log1pexp


class TestToyData(unittest.TestCase):

    def test_toy_data(self):

        parameters = Parameters()
        parameters.max_iter = 10000
        parameters.min_loss = 1e-3
        parameters.batch_size = 505
        parameters.step_size = 1e-2
        parameters.l1_W = 0

        k = 50  # number of components
        t = 20000  # number of time-steps
        d = 1000  # dimensionality of components

        X, H_true, W_true = self._generate_toydata(t, d, k)
        H, W, log = nmf(X, k, parameters)
        W_diff = np.linalg.norm(W-W_true)/t/d
        H_diff = np.linalg.norm(H-H_true)/t/d

        assert W_diff < 1e-4, "Optimization did not converge"
        assert H_diff < 1e-4, "Optimization did not converge"

    def _generate_toydata(self, t, d, k):

        H = log1pexp(np.random.randn(t, k))
        W = log1pexp(np.random.randn(k, d))

        X0 = H@W
        noise = np.random.randn(t, d)*0.001
        X = np.clip(X0 + noise, a_min=0, a_max=1)
        return np.array(X), np.array(H), np.array(W)
