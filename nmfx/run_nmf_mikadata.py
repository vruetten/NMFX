import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib as mpl
import sys, os
from glob import glob
from importlib import reload
sys.path.append('/groups/ahrens/home/ruttenv/code/zfish/')
from zfish.util import filesys as fs
from zfish.ephys import ephys as eph
from zfish.image import imclass as im
from importlib import reload
from zfish.models import _mynmf as myNMF
pl.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (21,4)




if __name__ == '__main__':
    base_dir = sys.argv[1]
    exps = list(fs.get_subfolders(base_dir).keys())[1:]
    exp_num = int(sys.argv[2])
    ch_num = int(sys.argv[3]) 

    knmf = int(sys.argv[4]) 
    l1_ratio = float(sys.argv[5])
    alpha_W = float(sys.argv[6])
    alpha_H = float(sys.argv[7])
    
    fnum = int(base_dir.split('_f')[1].split('_')[0])
    print('fnum: {}'.format(fnum))
    
    f = exps[exp_num]
    print('processing ' + f)
    folder_name = base_dir + f + '/'
    dirs = fs.get_subfolders(folder_name)   
    dpath = dirs['ephys'] + 'valid_cell_inds.npy'
    save_path = dirs['factors']

    print('\nknmf:{}, l1 ratio: {}, alpha_w: {}, alpha_h: {}'.format(knmf, l1_ratio, alpha_W, alpha_H))


    val_dict = np.load(dpath, allow_pickle = True).item()
    valid_inds = val_dict['valid_inds']
    tmax = val_dict['tvalidmax']
    print('loaded valid cells')

    mk = im.Mika(dirs = dirs, ch = ch_num)
    dims = mk.dims
    print('loading data...')
    mk.load_celldata()

    data = mk.df[valid_inds]
    # data = np.random.randn(100, 80)
    n, t = data.shape
    print('n: {}, t: {}'.format(n, t))


    data_pos = data - data.min() + 0.1
    # data_pos = demodata - demodata.min() + 0.1
    data_pos = data_pos[:,::2]
    # print(data_pos.shape)
    max_iter = 500