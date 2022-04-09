import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from jax.flatten_util import ravel_pytree
import optax


def compute_loss(params, X, l1_loss_weight):

    W = params['W']
    H = params['H']
    d, k = W.shape
    W_pos = sigmoid(W)
    H_pos = sigmoid(H)

    reconstruction = W_pos @ H_pos
    # print('test')
    reconstruction_loss = ((reconstruction - X)**2).mean()
    # print('test2')
    l1_loss = jnp.abs(H_pos).sum() * l1_loss_weight/(d*k)

    return reconstruction_loss + l1_loss


def sigmoid(Z):
    A=1/(1+(jnp.exp((-Z))))
    return A


def batch_update_step(params, X, batch_size, l1_loss_weight):

    # print("Updating!!!")
    W = params['W']
    H = params['H']

    t, d = X.shape

    # create random batch
    indices = np.random.randint(0, t, size=batch_size)
    non_indices = np.array([i for i in np.arange(t) if i not in indices])
    
    # batch_X = X[indices]
    # batch_W = W[indices]
    batch_X = X
    # params['W'] = batch_W

    loss, grad = jax.value_and_grad(compute_loss)(
        params,
        batch_X,
        l1_loss_weight
    )
    grad['W'].at[non_indices].set(0)
    return loss, grad



def update_step(W, H, X, batch_size, l1_loss_weight, step_size):

    print("Updating!!!")

    t, d = X.shape

    # create random batch
    indices = np.random.randint(0, t, size=batch_size)
    batch_X = X[indices]
    batch_W = W[indices]

    loss, grad = jax.value_and_grad(compute_loss)(
        (batch_W, H),
        batch_X,
        l1_loss_weight
    )

    W_delta, H_delta = grad[0], grad[1]


    W = W.at[indices,:].set(batch_W - W_delta * step_size)
    H = H - H_delta * step_size

    loss = loss
    return W, H, loss



def generate_toydata(t, d, k):

    H = sigmoid(np.random.randn(t, k))
    W = sigmoid(np.random.randn(k, d))
    
    X0 = H@W
    noise = np.random.randn(t, d)*0.001
    
    X = np.clip(0, 1, X0 + noise)

    return X, H, W


def initialize(X, k):
    t, d = X.shape
    W = jnp.ones((t, k))
    H = jnp.ones((k, d))
    return W, H
