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
from nmfx.old_utils.plot_funcs import *
from matplotlib import pyplot as pl
pl.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (21,4)


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
    batch_size = 50
    step_size = 1e-2
    print_iter = 100
    save_iter = 50


    date = datetime.today().strftime('%y%m%d%H%M%S')
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
    

    ### find W, H such that |X - W@H|^2 is minimized

    update_step = jax.jit(update_step, static_argnums = (3,4,5))

    t0 = time()
    for i in range(max_iterations):
        W, H, loss = update_step(W, H, X, batch_size, l1_loss_weight, step_size)
        losses.append(float(loss))
        if i % print_iter == 0:
            t1 = time()
            tdiff = np.round(t1-t0,3)
            print(f"Iteration {i}, loss={loss}, time={tdiff}")
        if loss < min_loss:
            print("Fitting converged!")
            break

    losses = np.array(losses)
    
    ### STORE EXPERIMENTAL RESULTS
    results['W_'] = W
    results['H_'] = H
    results['W'] = sigmoid(W)
    results['H'] = sigmoid(H)
    results['losses'] = losses
    
    np.save(results_path, results)
    print('results saved')

    plot_loss(results_path, losses)
    print('results plotted')
    
