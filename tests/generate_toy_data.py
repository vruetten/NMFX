import sys, os
import numpy as np

sys.path.append("/groups/ahrens/home/ruttenv/python_packages/nmfx/")

from nmfx.utils import log1pexp

np.random.seed(0)

t = 50
k = 9
coords_dim = 2
dim = 25


def generate_toy_data(dim, k, t):
    xv, yv = np.meshgrid(np.arange(dim) / dim, np.arange(dim) / dim)
    coords = np.vstack([xv.flatten()[None], yv.flatten()[None]]).T
    taus = np.linspace(0.001, 0.2, k)[::-1]
    taus[2:] = 1e-5
    dist = ((coords[None] - coords[:, None]) ** 2).sum(-1)
    dist_coef = [
        np.exp(-dist / tau) + np.diag(np.arange(dim * dim)) * 1e-4 for tau in taus
    ]
    Wkd = log1pexp(
        np.array(
            [
                np.random.multivariate_normal(np.zeros(dim * dim), K, size=1)
                for K in dist_coef
            ]
        )
    ).squeeze()
    H = log1pexp(np.random.randn(t, k))
    X = H @ Wkd + log1pexp(np.random.randn(t, dim * dim) * 0.0001)
    return X, H, Wkd, taus, coords


X, H, Wkd, taus, coordinates = generate_toy_data(dim, k, t)

data = {}
data["X"] = X
data["H"] = H
data["Wkd"] = Wkd
data["taus"] = taus
data["k"] = k
data["coordinates"] = coordinates
save_path = "/groups/ahrens/ahrenslab/Virginia_dm11/nmf_test/data.npy"
np.save(save_path, data)
print("data saved")
