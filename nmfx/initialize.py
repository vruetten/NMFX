import jax.numpy as jnp


def initialize(X, k, how = 'ones'):
    t, d = X.shape
    if how == 'ones':
        H = jnp.ones((t, k))
        W = jnp.ones((k, d))
    return H, W
