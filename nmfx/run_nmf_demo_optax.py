import jax
import jax.numpy as jnp
import numpy as np
import os, sys, getopt
from datetime import datetime
from time import time
sys.path.append('/Users/ruttenv/Documents/projects/nmfx/')
sys.path.append('/groups/ahrens/home/ruttenv/python_packages/nmfx/')
from nmf_funcs import *
from nmfx.utils.plot_funcs import *
from matplotlib import pyplot as pl
import optax
from functools import partial
from jax import jit



if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,'k:l:',['knum=','l1_loss_weight='])
    except getopt.GetoptError:
        print ('run_nmf.py -k <knum> -l <l1_loss_weight>')
        sys.exit(2)

    print('\n')
    for opt, arg in opts:
        if opt == '-k':
            k = int(arg)
            print(f'k: {k}')
        if opt in ('-l', '--l1_loss_weight'):
            l1_loss_weight = float(arg)
            print(f'l1_loss_weight: {l1_loss_weight}')
    print('\n')

    # number of components
    max_iterations = 10000
    min_loss = 1e-4
    batch_size = 16
    step_size = 1e-2
    print_iter = 10

    results_folder = '/Users/ruttenv/Documents/projects/nmfx/results/'
    results_folder = '/nrs/ahrens/Virginia_nrs/nmf_test'
    date = datetime.today().strftime('%y%m%d%H%M%S')
    experiment_name = date + f'_results_k_{k}_l1_loss_weight_{l1_loss_weight}'
    subfolder_path = results_folder +  experiment_name
    os.mkdir(subfolder_path)

    results_path = subfolder_path + '/' + experiment_name
    

    losses = []
    # X = load_data()
    t = 1000
    d = 80

    X, H_true, W_true = generate_toydata(t, d, k)
    W, H = initialize(X, k)

    results = {}
    results['W_init'] = W
    results['H_init'] = H    
    results['k'] = k
    results['max_iterations'] = max_iterations
    results['min_loss'] = min_loss
    results['batch_size'] = batch_size
    results['l1_loss_weight'] = l1_loss_weight
    results['step_size'] = step_size
    results['date'] = date

    initial_params = {
       'W': jnp.asarray(W),
       'H': jnp.asarray(H)
    } 

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.0,
        warmup_steps=50,
        decay_steps=1_000,
        end_value=0.0,
        )

    ### find W, H such that |X - W@H|^2 is minimized
    # update_step = jax.jit(update_step_optax, static_argnums = (3,4,5))

    t0 = time()
    def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
        opt_state = optimizer.init(params)
        print('starting')

        @partial(jit, static_argnums=(3,4,5))  
        def update_step_optax(params, opt_state, X, batch_size, l1_loss_weight):
            loss, grads = batch_update_step(params, X, batch_size, l1_loss_weight)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss


        for i in range(max_iterations):
            params, opt_state, loss = update_step_optax(params, opt_state, X, batch_size, l1_loss_weight)
            losses.append(float(loss))
            if i % print_iter == 0:
                t1 = time()
                tdiff = np.round(t1-t0,2)
                print(f"Iteration {i}, loss={np.round(loss,4)}, time={tdiff}min")
            if loss < min_loss:
                print("Fitting converged!")
                break

        return params

    

    optimizer = optax.chain(optax.adamw(learning_rate=schedule))
    params = fit(initial_params, optimizer) 
    

    losses = np.array(losses)
    
    ### store results
    results['W_'] = W
    results['H_'] = H
    results['W'] = sigmoid(W)
    results['H'] = sigmoid(H)
    results['losses'] = losses
    
    np.save(results_path, results)
    print('results saved')

    plot_loss(results_path, losses)
    print('results plotted')
    

  