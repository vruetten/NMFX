import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from nmfx.nmf_funcs import sigmoid
from .parameters import Log
from .initialize import initialize
import optax
from .losses import compute_batch_H_loss
from time import time
from .losses import compute_W_loss
from .updates import update_W_step
from .updates import update_W_batch_H_step
from jax import random



def nmf(X, k, parameters):

    print_iter = 50

    t, d = X.shape

    H, W = initialize(X, k, 'ones')

    log = Log()

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=50,
        decay_steps=1_000,
        end_value=0.0,
        )

    optimizer_W = optax.adamw(learning_rate=schedule)
    opt_state_W = optimizer_W.init(W)

    optimizer_H = optax.adamw(learning_rate=schedule)
    opt_state_H = optimizer_H.init(H)


    update_W_batch_H_step_jit = jax.jit(update_W_batch_H_step, static_argnames=['optimizer_W', 'total_batch_num', 'parameters'])
    
    total_batch_num = np.int(np.round(t/parameters.batch_size))
    print(f'total batch num: {total_batch_num}')
    
    
    t0 = time()

    key = random.PRNGKey(42)
    for i in range(parameters.max_iter):
        key, shuffle_key = random.split(key)
        
        W, opt_state_W, grad_H, loss_batch = update_W_batch_H_step_jit(X, H, W, optimizer_W, opt_state_W, opt_state_H, parameters, total_batch_num, shuffle_key)
            
        log.total_loss.append(loss_batch)
        
        updates_H, opt_state_H = optimizer_H.update(grad_H, opt_state_H, H)
        H = optax.apply_updates(H, updates_H)

        if i % print_iter == 0:
            t1 = time()
            tdiff = np.round(t1-t0,2)
            print(f"Iteration {i}, loss={np.round(loss_batch,4)}, time={tdiff}sec")
        if loss_batch < parameters.min_loss:
            print("Fitting converged!")
            break

    H = sigmoid(H)
    W = sigmoid(W)
    
    return H, W





