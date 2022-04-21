import jax.numpy as jnp


def sigmoid(Z):
    A=1/(1+(jnp.exp((-Z))))
    return A