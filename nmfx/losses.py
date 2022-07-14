import jax.numpy as jnp
from .parameters import IterationLog
from .utils import log1pexp
from .utils import sigmoid

def compute_W_loss(W, batch_X, batch_H, l1_W):

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X)/t/d

    l1_loss = jnp.abs(W_pos).mean() * l1_W
    total_loss = reconstruction_loss + l1_loss

    iteration_log = IterationLog()
    iteration_log.l1_loss_W = l1_loss.astype(float)
    iteration_log.reconstruction_loss = reconstruction_loss.astype(float)
    iteration_log.total_loss = total_loss.astype(float)

    return (total_loss, iteration_log)

def compute_batch_H_loss(batch_H, batch_X, W, l1_W):

    k, d = W.shape
    t, _ = batch_X.shape
    
    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)
    
    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X)/t/d

    l1_loss = jnp.abs(W_pos).mean() * l1_W
    total_loss = reconstruction_loss + l1_loss

    iteration_log = IterationLog()
    iteration_log.l1_loss_W = l1_loss.astype(float)
    iteration_log.reconstruction_loss = reconstruction_loss.astype(float)
    iteration_log.total_loss = total_loss.astype(float)
    return (total_loss, iteration_log)
