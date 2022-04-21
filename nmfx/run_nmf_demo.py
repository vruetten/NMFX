import jax
import jax.numpy as jnp
import numpy as np
import os, sys, getopt
from datetime import datetime
from time import time
sys.path.append('/Users/ruttenv/Documents/projects/nmfx/')
sys.path.append('/groups/ahrens/home/ruttenv/python_packages/nmfx/')
from nmf_funcs import *
from nmfx.old_utils.plot_funcs import *
from matplotlib import pyplot as pl
import optax
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative




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
    max_iterations = 5000
    min_loss = 1e-4
    batch_size = 16
    step_size = 1e-2
    print_iter = 500

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

    ### find W, H such that |X - W@H|^2 is minimized

    update_step = jax.jit(update_step, static_argnums = (2,3,4))

    t0 = time()
    params = initial_params

    
    for i in range(max_iterations):
        W, H, loss = update_step(params, X, batch_size, l1_loss_weight, step_size)
        losses.append(float(loss))
        if i % print_iter == 0:
            t1 = time()
            tdiff = np.round(t1-t0,3)
            print(f"Iteration {i}, loss={loss}, time={tdiff}")
        if loss < min_loss:
            print("Fitting converged!")
            break

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
    

  