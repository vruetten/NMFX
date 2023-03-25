import jax.numpy as jnp
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
import numpy as np

######## KERNEL FUNCTIONS  #########

sq_fplus = lambda tau, l: jnp.exp(-1 / 2 * (tau**2).sum(-1) / (l**2))


def compute_coord_dist(x0: jnp.array, x1: jnp.array) -> jnp.array:
    return x0[None, :, :] - x1[:, None]


def build_K(x0: jnp.array, x1: jnp.array, param) -> jnp.array:
    """Builds kernel

    Parameters:
    x0: N x dim
    x1: M x dim
    dim: number of coordinates
    param: parameter of the kernel
    fplus: kernel function

    Returns:
    K

    """
    tau = compute_coord_dist(
        x0, x1
    )  # computes euclidean distance between all points N x N x dim
    K = sq_fplus(tau, param)
    return K


def compute_wKiw(Kxx: jnp.array, w: jnp.array):
    """Compute wKiw

    K = LL^T where L is lower triangular matrix

    w^TKiw = (Liw)^T(Liw)
    Let: z = Liw
    Then: Lz = w

    """
    L = cholesky(Kxx)
    z = solve_triangular(L, w, lower=True)
    wKiw = z.T @ z
    return wKiw


def sample_from_K(Kxx: jnp.array) -> np.array:
    import numpy as np
    n = Kxx.shape[0]
    L = cholesky(Kxx)
    x = np.random.randn(n)
    return L.T @ x
