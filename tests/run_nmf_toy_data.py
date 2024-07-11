import sys, os, re

sys.path.append("/groups/ahrens/home/ruttenv/python_packages/nmfx/")
import numpy as np
from nmfx.parameters import Parameters
from nmfx.utils import log1pexp, logexpm1
import nmfx
from jax.lib import xla_bridge
from time import time


print(xla_bridge.get_backend().platform)
print("\n")

save_path = "/groups/ahrens/ahrenslab/Virginia_dm11/nmf_test/data.npy"
data = np.load(save_path, allow_pickle=True).item()

H = data["H"]
Wkd = data["Wkd"]
X = data["X"]
coordinates = data["coordinates"]
k = data["k"]
taus = np.linspace(0.001, 0.2, k)[::-1]
taus[2:] = 0
taus[:3] = 0.2
taus = data["taus"][::-1]

t, d = X.shape
save_iter = 2000

initial_values = {}
initial_values["H"] = H
initial_values["W"] = Wkd

parameters = Parameters()
parameters.batch_size = t  # no batches - deterministic
parameters.max_iter = 50000
parameters.l1_W = 0
l2_params = np.arange(start=0, stop=0.005, step=0.0002)
# l2_params = [0, 0.00]
l2_params = l2_params[:2]
for l2 in l2_params:
    t0 = time()
    print(f"running {l2}")
    save_path = f"/groups/ahrens/ahrenslab/Virginia_dm11/nmf_test/l2_{l2}/"
    save_path = re.sub(r"([.?!]+) *", r"p", save_path)
    os.makedirs(save_path, exist_ok=True)
    parameters.l2_space = l2
    parameters.l1_W = 0
    parameters.step_size = 1e-2
    # coordinates = None

    H_, W_, log = nmfx.nmf(
        X,
        k,
        parameters,
        taus=taus,
        coordinates=coordinates,
        save_path=save_path,
        save_iter=save_iter,
        print_iter=save_iter,
        init="random",
    )
    t1 = np.round((time() - t0) / 60, 3)
    print(f"time: {t1}\n\n\n\n\n")
