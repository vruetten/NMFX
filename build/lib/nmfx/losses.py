import jax.numpy as jnp
from .utils import sigmoid
from .utils import log1pexp

def compute_W_loss(W, batch_X, batch_H, l1_W):

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X)/t/d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    return reconstruction_loss + l1_loss


def compute_batch_H_loss(batch_H, batch_X, W, l1_W):

    k, d = W.shape
    t, _ = batch_X.shape
    
    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X)/t/d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    return reconstruction_loss + l1_loss
