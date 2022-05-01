import jax.numpy as jnp
import jax
import numpy as np
from nmfx import sigmoid
from nmfx import nmf
from nmfx.parameters import Parameters
import os

def generate_toydata(t, d, k):

    H = sigmoid(np.random.randn(t, k))
    W = sigmoid(np.random.randn(k, d))
    
    X0 = H@W
    noise = np.random.randn(t, d)*0.01
    X = X0 + noise
    
    # X = np.clip(0, 1, X0 + noise*0)

    return X, H, W


if __name__ == '__main__':

    print("Device:", jax.default_backend())

    parameters = Parameters()
    parameters.max_iter = 10000
    parameters.min_loss = 1e-3
    parameters.batch_size = 505
    parameters.step_size = 1e-2
    parameters.l1_W = 0

    k = 50

    t = 2000
    d = 4000

    data_path = '/nrs/ahrens/Virginia_nrs/nmfx_tests/data.npy'
    results_path = '/nrs/ahrens/Virginia_nrs/nmfx_tests/results.npy'
    

    X, H_true, W_true = generate_toydata(t, d, k)
    np.save(data_path, X)

    # cmd = f'nmfx --data_path {data_path} --save_path {results_path} -k {k} '
    cmd = f'python ./scripts/nmfx --data_path {data_path} --save_path {results_path} -k {k} '
    print(cmd)
    os.system(cmd)

    results = np.load(results_path, allow_pickle = True).item()
    print(W_true.shape, results['W'].shape)

    W_diff = np.linalg.norm(results['W']-W_true)/t/d
    H_diff = np.linalg.norm(results['H']-H_true)/t/d
    print(f'l2 norm W diff per data pt:{np.round(W_diff,5)}')
    print(f'l2 norm H diff  per data pt: {np.round(H_diff,5)}')





