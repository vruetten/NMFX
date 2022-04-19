import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as pl
import matplotlib as mpl
import sys, os
from glob import glob
sys.path.append('/groups/ahrens/home/ruttenv/code/zfish/')
sys.path.append('/groups/ahrens/home/ruttenv/python_packages/nmfx/')
from zfish.util import filesys as fs
from zfish.ephys import ephys as eph
from zfish.image import imclass as im
import getopt
from datetime import datetime
from time import time
from nmf_funcs import *
from nmfx.utils.plot_funcs import *
from matplotlib import pyplot as pl
import optax
from functools import partial
from jax import jit
pl.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (21,4)



def save_results(results_path, W, H, losses_H):

    losses_H = np.array(losses_H)
    # losses_H = losses_H.reshape([-1, total_batch_num])
     
    ### store results
    results['W_'] = W
    results['H_'] = H
    results['W'] = sigmoid(W)
    results['H'] = sigmoid(H)
    results['losses'] = losses_H
    
    np.save(results_path, results)
    print('results saved')

    plot_loss(results_path, losses_H)
    print('results plotted')
 

if __name__ == '__main__':

    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv,'i:e:c:k:l:',['input=','exp=','ch=','knum=','l1_loss_weight='])
    except getopt.GetoptError:
        print ('run_nmf_mika.py -i <input_path> -e <exp> -c <ch> -k <knum> -l <l1_loss_weight>')
        sys.exit(2)
    
    print('\nparameters:')
    for opt, arg in opts:
        if opt in ('-i', '--input'):
            base_dir = arg
        if opt in ('-e', '--exp'):
            exp_num = int(arg)
            print(f'expnum: {exp_num}')
        if opt in ('-c', '--ch'):
            ch_num = int(arg)
            print(f'ch: {ch_num}')
        if opt in ('-k', '--knum'):
            k = int(arg)
            print(f'k: {k}')
        if opt in ('-l', '--l1_loss_weight'):
            l1_loss_weight = float(arg)
            print(f'l1_loss_weight: {l1_loss_weight}')
    print('\n')

    exps = list(fs.get_subfolders(base_dir).keys())[1:]    
    fnum = int(base_dir.split('_f')[1].split('_')[0])
    exp = exps[exp_num]
    

    print('fnum: {}'.format(fnum))
    print('processing ' + exp)
    folder_name = base_dir + exp + '/'
    dirs = fs.get_subfolders(folder_name)   
    dpath = dirs['ephys'] + 'valid_cell_inds.npy'
    save_path = dirs['factors']

    print('\nknmf:{}, l1 loss: {}'.format(k, l1_loss_weight))


    val_dict = np.load(dpath, allow_pickle = True).item()
    valid_inds = val_dict['valid_inds']
    tmax = val_dict['tvalidmax']
    print('loaded valid cells')

    t0 = time()
    mk = im.Mika(dirs = dirs, ch = ch_num)
    dims = mk.dims
    print('loading data...')
    mk.load_celldata()
    t1 = time()
    tdiff = np.round(t1-t0)/60
    print(f'loaded data in {tdiff}mins')

    data = mk.df[valid_inds]
    n, t = data.shape
    print('n: {}, t: {}'.format(n, t))

    
    data_pos = data - data.min() + 0.1 # make data positive
    data_pos /=data_pos.max() # normalize data between 0 and 1
    data_pos = data_pos # subsample data

    max_iterations = 10000
    min_loss = 1e-4
    batch_size = 250
    step_size = 1e-2
    print_iter = 100
    save_iter = 100


    date = datetime.today().strftime('%y%m%d_%H%M%S')
    experiment_name = date + f'_results_k_{k}_l1_loss_weight_{l1_loss_weight}'
    subfolder_path = save_path +  experiment_name
    os.mkdir(subfolder_path)

    results_path = subfolder_path + '/' + experiment_name
    

    losses = []

    X = data_pos.T
    W, H = initialize(X, k)

    ### STORE EXPERIMENTAL PARAMETERS ###
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

    def update_H_step(params_H, opt_state_H, batch_X, batch_W, l1_loss_weight):
            # params only contains H

            loss_H, grad_H = jax.value_and_grad(compute_H_loss)(
                params_H,
                batch_X,
                batch_W,
                l1_loss_weight
            )

            updates, opt_state_H = optimizer_H.update(grad_H, opt_state_H, params_H)
            params_H = optax.apply_updates(params_H, updates)
            return params_H, opt_state_H, loss_H

    update_H_step_jit = jax.jit(update_H_step, static_argnums = (5))

    # outer loop
    params_H = {}
    params_H['H'] = H
    params_W = {}
    total_batch_num = t//batch_size
    print(f'total batch num: {total_batch_num}')
    params_W['W'] = W[:total_batch_num*batch_size]

    params_W_batch = {}

    optimizer_H = optax.chain(optax.adamw(learning_rate=schedule))
    opt_state_H = optimizer_H.init(params_H)

    optimizer_W = optax.chain(optax.adamw(learning_rate=schedule))
    opt_state_W = optimizer_W.init(params_W)

    indices = np.arange(total_batch_num*batch_size)

    losses_H = []

    t0 = time()
    for i in range(max_iterations):
        # shuffle all indices and chunk data into batches
        shuffled_indices = np.arange(total_batch_num*batch_size)
        np.random.shuffle(shuffled_indices)
        shuffled_batch_indices = shuffled_indices.reshape([-1, batch_size])

        # inner loop
        grad_W_batches = []
        for j in range(total_batch_num):
            batch_indices = shuffled_batch_indices[j]
            batch_X = X[batch_indices]
            batch_W = W[batch_indices]
            params_W_batch['W'] = batch_W

            # get batch_W grad using batch_X and H
            loss_W_batch, grad_W_batch = jax.value_and_grad(compute_batch_W_loss)(
                params_W_batch,
                batch_X,
                H,
                l1_loss_weight
            )

            # update H using batch_X and batch_W
            params_H, opt_state_H, loss_H = update_H_step_jit(params_H, opt_state_H, batch_X, batch_W, l1_loss_weight)
            losses_H.append(loss_H)
            grad_W_batches.append(grad_W_batch['W'])

        grad_W_batches = np.vstack(grad_W_batches)
        grad_W = {}
        grad_W['W'] = grad_W_batches[np.argsort(shuffled_indices).squeeze()]

        # update W
        updates_W, opt_state_W = optimizer_W.update(grad_W, opt_state_W, params_W)
        params_W = optax.apply_updates(params_W, updates_W)

        if i % print_iter == 0:
            t1 = time()
            tdiff = np.round(t1-t0,2)
            print(f"Iteration {i}, loss={np.round(loss_H,4)}, time={tdiff}sec")
        if loss_H < min_loss:
            print("Fitting converged!")
            save_results(results_path + f'_ite{i}', params_W['W'], params_H['H'], losses_H)
            break
        if i % save_iter == 0:
            save_results(results_path + f'_ite{i}', params_W['W'], params_H['H'], losses_H)

