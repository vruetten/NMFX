import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from jax.flatten_util import ravel_pytree



def compute_loss(W_and_H, X, l1_loss_weight):

    W, H = W_and_H
    d, k = W.shape
    W_pos = sigmoid(W)
    H_pos = sigmoid(H)

    reconstruction = W_pos @ H_pos
    
    reconstruction_loss = ((reconstruction - X)**2).mean()

    l1_loss = jnp.abs(H_pos).sum() * l1_loss_weight/(d*k)

    return reconstruction_loss + l1_loss


def sigmoid(Z):
    A=1/(1+(jnp.exp((-Z))))
    return A


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
    
    # flat_grad, unflatten = ravel_pytree(grad)
    # flat_adam_grad = adam_update_step(flat_grad, step_size)
    # adam_grad = unflatten(flat_adam_grad)

    W_delta, H_delta = grad[0], grad[1]


    W = W.at[indices,:].set(batch_W - W_delta * step_size)
    H = H - H_delta * step_size

    loss = loss
    return W, H, loss



def adam_update_step(flat_grad, step_size, m, v, i):

    g = flat_grad
    b1=0.9
    b2=0.999
    eps=10**-8

    m = (1 - b1) * g      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))    # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    adam_gradient = step_size*mhat/(np.sqrt(vhat) + eps)

    return adam_gradient


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
