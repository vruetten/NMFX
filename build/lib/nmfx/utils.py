import jax.numpy as jnp


log1pexp = lambda x: jnp.log1p(jnp.exp(x))
logexpm1 = lambda x: jnp.log(jnp.expm1(x))


def sigmoid(Z):
    A=1/(1+(jnp.exp((-Z))))
    return A


def order_factors(H, W):
    import numpy as np
    from numpy import argsort
    Hnorm = np.linalg.norm(H, axis = 0)
    Wnorm = np.linalg.norm(W, axis = 1)
    # W /= Wnorm[:,None]
    # H *= Wnorm[None]
    Knorm = Hnorm*Wnorm
    order = argsort(Knorm)[::-1]
    H = H[:,order]
    W = W[order]
    return H, W


def get_subfolders(folder, extension = '*', verbose = False, make_plot_folder = False, windows = False):
    from glob import glob
    import os
    ''' extract subfolders and paths of a main directory containing extension'''
    
    folder_paths = list(filter(lambda v: os.path.isdir(v), sorted(glob(folder+ extension))))
    if windows == True:  div = '\\'
    else: div = '/'
    folders = list(map(lambda v: v.split(div)[-1], folder_paths))
    dirs = {}
    dirs['main'] = folder
    for i in range(len(folders)):
        if os.path.isdir(folder_paths[i]):
            dirs[folders[i]] = folder_paths[i] + div
    if verbose:
        [print(k) for ind, k in enumerate(dirs.keys())]
    return dirs