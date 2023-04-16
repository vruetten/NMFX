import sys, os

sys.path.append("/groups/ahrens/home/ruttenv/python_packages/nmfx/")
import numpy as np
from nmfx.parameters import Parameters
from nmfx.utils import log1pexp, logexpm1
import nmfx


save_path = "/groups/ahrens/ahrenslab/Virginia_dm11/nmf_test/data.npy"
data = np.load(save_path, allow_pickle=True).item()

H = data["H"]
Wkd = data["Wkd"]
X = data["X"]
coordinates = data["coordinates"]
taus = data["taus"][::-1]
k = data["k"]
t, d = X.shape
save_iter = 400

initial_values = {}
initial_values["H"] = H
initial_values["W"] = Wkd

parameters = Parameters()
parameters.batch_size = t
parameters.max_iter = 5000
parameters.l1_W = 0

l2_params = np.arange(start=0, stop=0.5, step=0.1)
for l2 in l2_params:
    print(f"running {l2}")
    save_path = f"/groups/ahrens/ahrenslab/Virginia_dm11/nmf_test/l2_{l2*10}/"
    os.makedirs(save_path, exist_ok=True)
    parameters.l2_space = l2

    H_, W_, log = nmfx.nmf(
        X,
        k,
        parameters,
        taus=taus,
        coordinates=coordinates,
        save_path=save_path,
        save_iter=save_iter,
    )
