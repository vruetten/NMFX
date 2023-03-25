import jax.numpy as jnp
from .utils import sigmoid, log1pexp
import numpy as np


def initialize(X: np.array, k: np.int, how="random"):
    t, d = X.shape
    if how == "random":
        H = log1pexp(np.random.randn(t, k) + 2) 
        W = log1pexp(np.random.randn(k, d) + 2)
    return H, W


def initialize_taus(k: np.int):
    taus = np.random.uniform(low=1e-3, high=120, size=k)
    return taus
