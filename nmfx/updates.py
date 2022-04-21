import optax
import jax
from .losses import compute_W_loss

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