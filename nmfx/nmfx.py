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

    def update_W_step(W, opt_state_W, batch_X, batch_H, l1_W):
            loss_W, grad_W = jax.value_and_grad(compute_W_loss)(
                W,
                batch_X,
                batch_H,
                l1_W
            )
            updates, opt_state_W = optimizer_W.update(grad_W, opt_state_W, W)
            W = optax.apply_updates(W, updates)
            return W, opt_state_W, loss_W

    update_W_step_jit = jax.jit(update_W_step)
    
    total_batch_num = np.int(np.round(t/parameters.batch_size))
    print(f'total batch num: {total_batch_num}')
    

    shuffled_indices = np.arange(t)
    t0 = time()
    for i in range(parameters.max_iter):
        # shuffle all indices and chunk data into batches
        np.random.shuffle(shuffled_indices)
        # inner loop
        grad_H_batches = []     
        for j in range(total_batch_num):
            batch_indices = shuffled_indices[j*parameters.batch_size:(j+1)*parameters.batch_size]
            # batch_indices = shuffled_batch_indices[j]
            batch_X = X[batch_indices]
            batch_H = H[batch_indices]

            # get batch_H grad using batch_X and H
            grad_H_batch = jax.grad(compute_batch_H_loss)(
                batch_H,
                batch_X,
                W,
                parameters.l1_W
            )

            # update H using batch_X and batch_H
            W, opt_state_W, loss_batch = update_W_step_jit(W, opt_state_W, batch_X, batch_H, parameters.l1_W)
            
            log.total_loss.append(loss_batch)
            grad_H_batches.append(grad_H_batch)

        grad_H_batches = np.vstack(grad_H_batches)
        grad_H = grad_H_batches[np.argsort(shuffled_indices).squeeze()]
        
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





