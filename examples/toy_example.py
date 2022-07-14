import jax.numpy as jnp
import numpy as np
from nmfx import nmf
from nmfx.parameters import Parameters
from nmfx.utils import sigmoid, log1pexp


def generate_toydata(t, d, k):

    H = sigmoid(np.random.randn(t, k))
    W = sigmoid(np.random.randn(k, d))
    
    X0 = H@W
    noise = np.random.randn(t, d)*0.01
    
    X = np.clip(0, 1, X0 + noise)

    return X, H, W

def generate_toydata(t, d, k):
    from nmfx.utils import sigmoid, log1pexp
    H = log1pexp(np.random.randn(t, k))
    W = log1pexp(np.random.randn(k, d))
    
    X0 = H@W
    noise = np.random.randn(t, d)*0.001
    X = np.clip(X0 + noise, a_min=0, a_max=1)
    return np.array(X), np.array(H), np.array(W)


if __name__ == '__main__':

    parameters = Parameters()
    parameters.max_iter = 10000
    parameters.min_loss = 1e-3
    parameters.batch_size = 505
    parameters.step_size = 1e-2
    parameters.l1_W = 0

    k = 50

    # X = load_data()
    t = 20000
    d = 1000

    X, H_true, W_true = generate_toydata(t, d, k)
    H, W, log = nmf(X, k, parameters)
    W_diff = np.linalg.norm(W-W_true)/t/d
    H_diff = np.linalg.norm(H-H_true)/t/d
    print(f'l2 norm W diff {W_diff}')
    print(f'l2 norm H diff {H_diff}')





