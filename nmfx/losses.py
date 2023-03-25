import jax.numpy as jnp
from .utils import sigmoid
from .utils import log1pexp
from .kernels import build_K
from .kernels import compute_wKiw
from typing import Tuple


def compute_spatial_prior_loss(W, taus, coordinates) -> jnp.array:
    k, d = W.shape
    Ks = [build_K(coordinates, coordinates, tau) for tau in taus]
    wKiW = jnp.sum([compute_wKiw(Ks[i], W[i]) for i in range(k)]) / k
    return wKiW


def compute_spatial_loss_coefficients(taus, coordinates) -> jnp.array:
    coords_dim = coordinates.shape[-1]
    dist_coef = jnp.exp(
        -(
            (
                coordinates[
                    None,
                    None,
                ]
                - coordinates[None, :, None]
            )
            ** 2
        ).sum(-1)
        / taus[:, None, None]
        / coords_dim
    )
    return dist_coef


def compute_W_loss(W, batch_X, batch_H, l1_W, spatial_loss_coefficients=None):

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)  # force W to be positive
    batch_H_pos = log1pexp(batch_H)  # force W to be positive

    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X) / t / d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    loss = reconstruction_loss + l1_loss

    if spatial_loss_coefficients is not None:
        w_dist = (W_pos[:, None] - W_pos[:, :, None]) ** 2
        spatial_penalty = (spatial_loss_coefficients * w_dist).sum()
        loss += spatial_penalty

    return loss


def compute_batch_H_loss(batch_H, batch_X, W, l1_W, spatial_loss_coefficients=None):

    k, d = W.shape
    t, _ = batch_X.shape

    W_pos = log1pexp(W)
    batch_H_pos = log1pexp(batch_H)

    reconstruction = batch_H_pos @ W_pos

    reconstruction_loss = jnp.linalg.norm(reconstruction - batch_X) / t / d

    l1_loss = jnp.abs(W_pos).mean() * l1_W

    if spatial_loss_coefficients is not None:
        w_dist = (W_pos[:, None] - W_pos[:, :, None]) ** 2
        spatial_penalty = (spatial_loss_coefficients * w_dist).sum()
        loss += spatial_penalty

    return reconstruction_loss + l1_loss
