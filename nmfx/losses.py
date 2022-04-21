import jax.numpy as jnp
from .utils import sigmoid


def compute_W_loss(W, batch_X, batch_H, l1_W):

    k, d = W.shape
    W_pos = sigmoid(W)
    batch_H_pos = sigmoid(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = ((reconstruction - batch_X)**2).mean()

    l1_loss = jnp.abs(W_pos).sum() * l1_W/(d*k)

    return reconstruction_loss + l1_loss


def compute_batch_H_loss(batch_H, batch_X, W, l1_W):

    k, d = W.shape
    W_pos = sigmoid(W)
    batch_H_pos = sigmoid(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = ((reconstruction - batch_X)**2).mean()

    l1_loss = jnp.abs(W_pos).sum() * l1_W/(d*k)

    return reconstruction_loss + l1_loss
