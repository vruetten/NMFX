#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-05-01 21:04:12

@author: virginiarutten
@email: gin.rutten@gmail.com
"""

from autograd.misc import flatten
import jax.numpy as np
from toolz import curry

def optimise_params(gradfun, params, step_size, num_ite = 100, \
                    threshold = 1e-3, callback = None, \
                    hyperiter = -1):
    # recieves flattened input
    cbs = []
    # old_val, _ = gradfun(params, 0, in_callback = True)
    old_val = 0
    # xo, unflatten = flatten(params)

    xo = params
    # g0 = gradfun(xo, 0, False)
    # print(g0.shape)


    # _grad = lambda x, i: flatten(gradfun(unflatten(x), i, ))[0]
    _grad = lambda x, i: gradfun(x, i, False)
    
    # x = flatten(params)[0]
    x = params
    ##### parameters for adam ####
    b1=0.9
    b2=0.999
    eps=10**-8
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_ite):
        g = _grad(x, i)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        tmp = step_size*mhat/(np.sqrt(vhat) + eps)
        x -= tmp


        if i % 10 ==0:
            # val, grad_norm = gradfun(unflatten(x), 9, in_callback = True)  
            val, grad_norm = gradfun(x, 0, in_callback = True)  
            diff = old_val - val
            old_val = val
            print('diff: {:.6f}'.format(diff))
            if np.abs(diff)<threshold:
                break

        if callback:
            cb = callback(x, i, g, hyperiter)
            cbs.append(cb[0])
    
    return x, cbs




@curry
def print_perf(gradfun, eval_ite, print_ite, params, iter, gradient, hyperiter = -1, adam = True):
    val = None
    norm_grad = None
    if hyperiter == 0:
        print("     Epoch     |    Cost  |       Test accuracy  ")
    if hyperiter == -1: # i.e.: listen to iter
        if iter == 0:
            print("     Epoch     |    Cost  |       Test accuracy  ")
        if iter % eval_ite == 0:
            if adam:
                val, norm_grad = gradfun(params, 0, in_callback = True)
                # print(val)

        if iter % print_ite == 0:
            print("{:15}|{:20}|{:20}".format(iter, val, 0))
    else:
        if hyperiter % eval_ite == 0:
            val, norm_grad = gradfun(params, 0, in_callback = True)
        if hyperiter % print_ite == 0:
            if iter % 10 ==0:
                val, norm_grad = gradfun(params, 0, in_callback = True)
                print("{:15}|{:20}|{:20}".format(hyperiter, np.int(val), 0))
    return (val, norm_grad)


def convert_cbs_to_list(cbs):
    vals = np.array([i[0] for i in cbs])
    ngrad = [i[1] for i in cbs]
    vals = list(filter(None, vals))
    ngrad = list(filter(None, ngrad))
    return vals, ngrad
