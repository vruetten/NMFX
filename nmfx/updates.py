import optax
import jax
from .losses import compute_H_loss

def update_H_step(H, optimizer_H, opt_state_H, batch_X, batch_W, l1_W):
        loss_H, grad_H = jax.value_and_grad(compute_H_loss)(
            H,
            batch_X,
            batch_W,
            l1_W
        )
        updates, opt_state_H = optimizer_H.update(grad_H, opt_state_H, H)
        H = optax.apply_updates(H, updates)
        return H, opt_state_H, loss_H