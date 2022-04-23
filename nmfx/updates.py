import optax
import jax
from .losses import compute_W_loss
from .losses import compute_batch_H_loss
from jax import random
import jax.numpy as jnp

def update_W_step(W, optimizer_W, opt_state_W, batch_X, batch_H, l1_W):
        loss_W, grad_W = jax.value_and_grad(compute_W_loss)(
            W,
            batch_X,
            batch_H,
            l1_W
        )
        updates, opt_state_W = optimizer_W.update(grad_W, opt_state_W, W)
        W = optax.apply_updates(W, updates)
        return W, opt_state_W, loss_W


def update_W_batch_H_step(X, H, W, optimizer_W, opt_state_W, opt_state_H, parameters, total_batch_num, shuffle_key):
    print('compiling update function')
    t, d = X.shape
    
    shuffled_indices = jnp.arange(t)
    random.permutation(shuffle_key, shuffled_indices)

    grad_H_batches = [] 
    for j in range(total_batch_num):
        batch_indices = shuffled_indices[j*parameters.batch_size:(j+1)*parameters.batch_size]
        batch_X = X[batch_indices]
        batch_H = H[batch_indices]

        grad_H_batch = jax.grad(compute_batch_H_loss)(
            batch_H,
            batch_X,
            W,
            parameters.l1_W
        )

        W, opt_state_W, loss_batch = update_W_step(W, optimizer_W, opt_state_W, batch_X, batch_H, parameters.l1_W)
        grad_H_batches.append(grad_H_batch)

    grad_H_batches = jnp.vstack(grad_H_batches)
    grad_H = grad_H_batches[jnp.argsort(shuffled_indices).squeeze()]

    return W, opt_state_W, grad_H, loss_batch 
    