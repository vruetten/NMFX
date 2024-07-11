import jax.numpy as jnp
from .utils import sigmoid, log1pexp
import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf


def initialize(X: np.array, k, how="random"):
    t, d = X.shape
    print("initializing")
    if how == "random":
        print("randomly")
        H = log1pexp(np.random.randn(t, k) + 2)
        W = log1pexp(np.random.randn(k, d) + 2)
    if how == "nndsvd":
        print("with nnsvd")
        H, W = initialize_nmf(X, k, "nndsvd")
        print(H.shape, W.shape)
    return H, W


def initialize_taus(k):
    taus = np.random.uniform(low=1e-3, high=120, size=k)
    return taus
