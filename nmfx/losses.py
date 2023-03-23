import jax.numpy as jnp
from .utils import sigmoid
from .utils import log1pexp
from .kernels import build_K
from .kernels import compute_wKiw
from types import Tuple


def compute_spatial_prior_loss(W, taus, coordinates):
    k, d = W.shape
    Ks = [build_K(coordinates, coordinates, tau) for tau in taus]
    wKiW = jnp.sum([compute_wKiw(Ks[i], W[i]) for i in range(k)]) / k
    return wKiW


def compute_taus_loss(taus, W, X, H, l1_W, coordinates=None) -> jnp.float:

    k, d = W.shape
    t, _ = X.shape

    W_pos = log1pexp(W)
    H_pos = log1pexp(H)

    reconstruction = H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - X) / t / d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    loss = reconstruction_loss + l1_loss

    if taus is not None:
        wKiw = compute_spatial_prior_loss(W, taus, coordinates)
        loss += wKiw

    return loss


def compute_W_loss(W, batch_X, batch_H, l1_W, taus=None, coordinates=None) -> jnp.float:

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)

    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X) / t / d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    loss = reconstruction_loss + l1_loss

    if taus is not None:
        wKiw = compute_spatial_prior_loss(W, taus, coordinates)
        loss += wKiw

    return loss


def compute_batch_H_loss(batch_H, batch_X, W, l1_W, taus=None, coordinates=None):

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)

    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X) / t / d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    if taus is not None:
        wKiw = compute_spatial_prior_loss(W, taus, coordinates)
        loss += wKiw

    return reconstruction_loss + l1_loss
