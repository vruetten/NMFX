import jax
import jax.numpy as jnp
from jax import jit
from jax import random
import numpy as np
from time import time
import optax

from .utils import sigmoid, log1pexp, order_factors
from .parameters import Log
from .initialize import initialize
from .losses import compute_batch_H_loss
from .losses import compute_W_loss
from .updates import update_W_step
from .updates import update_W_batch_H_step


def nmf(X, k, parameters, print_iter = 20, initial_values = None):
    t, d = X.shape
    X_min = np.min(X)

    if X_min<0:
        print('Error! X must be positive :)')
        raise ValueError
        
    if parameters.batch_size>t:
        parameters.batch_size = t
        print('batch size greater than t - resetting batch size')
   
    if initial_values is None:
        H, W = initialize(X, k, 'random')
    else:
        H = initial_values['H']
        W = initial_values['W']
        print('H & W initialized with given initial values')

    

    log = Log()

    ### optimizers
    optimizer_W = optax.adam(learning_rate = parameters.step_size)
    opt_state_W = optimizer_W.init(W)

    optimizer_H = optax.adam(learning_rate = parameters.step_size)
    opt_state_H = optimizer_H.init(H)

    update_W_batch_H_step_jit = jax.jit(update_W_batch_H_step, static_argnames=['optimizer_W', 'total_batch_num', 'parameters'])
    

    H_prev = np.copy(H)
    total_batch_num = np.int(np.round(t/parameters.batch_size))
    print(f'total batch num: {total_batch_num}')
    
    t0 = time()

    key = random.PRNGKey(42)
    try:
        for i in range(parameters.max_iter):
            key, shuffle_key = random.split(key)
            
            W, opt_state_W, grad_H, loss_batch = update_W_batch_H_step_jit(X, H, W, optimizer_W, opt_state_W, opt_state_H, parameters, total_batch_num, shuffle_key)
            loss_batch = float(loss_batch) 
            grad_norm = np.linalg.norm(grad_H)
            log.total_loss.append(loss_batch)
            log.grad_norm.append(grad_norm)
            
            updates_H, opt_state_H = optimizer_H.update(grad_H, opt_state_H, H)
            H = optax.apply_updates(H, updates_H)
            H_diff = np.linalg.norm(H-H_prev)/k/t*1000
            H_prev = np.copy(H)

            if i % print_iter == 0:
                t1 = time()
                tdiff = np.round(t1-t0,2)
                print(f"Iteration {i}, loss={np.round(loss_batch,10)}, h_diff={np.round(H_diff,7)}, grad_norm={np.round(grad_norm,10)}, time={tdiff}sec")
                print(np.abs(updates_H).sum())
                print(np.abs(grad_H).max())
            if loss_batch < parameters.min_loss:
                print(f"Iteration {i}, loss={np.round(loss_batch,10)}, h_diff={np.round(H_diff,7)}, grad_norm={np.round(grad_norm,10)}, time={tdiff}sec")
                print("Fitting converged!")
                break
            if i>10:
                if H_diff < parameters.min_diff:
                    print(f"Iteration {i}, loss={np.round(loss_batch,10)}, h_diff={np.round(H_diff,7)}, grad_norm={np.round(grad_norm,10)}, time={tdiff}sec")
                    print("Fitting converged! Minimal difference reached.")
                    break

    except KeyboardInterrupt:
        print('caught Keyboard interrupt')

    ### project solution back into feasible space
    H = log1pexp(H)
    W = log1pexp(W)

    ### normalize and order factors by norm
    H, W = order_factors(H, W)
    
    return np.array(H), np.array(W), log





