import jax.numpy as jnp
import jax
import numpy as np
from nmfx import sigmoid
from nmfx import nmf
from nmfx.parameters import Parameters
import os


if __name__ == '__main__':

    print("Device:", jax.default_backend())

    parameters = Parameters()
    parameters.max_iter = 1000
    parameters.min_loss = 1e-5
    parameters.batch_size = 50
    parameters.step_size = 1e-2
    parameters.l1_W = 0
    parameters.min_diff = 1e-4 #checked

    k = 50
        
    script_path = '/groups/ahrens/home/ruttenv/python_packages/nmfx/scripts/nmfx_hdf5'
    folder_path = '/nrs/ahrens/Virginia_nrs/LS_sick/220214_f101_HuC_H2B_gCaMP7f_gfap_jRGECO1b_7dpf_planx_25ugml/'
    experiment_number = 0
    channel_number = 0
    gpu_cmd = 'bsub -J "GPUJob" -n 6 -gpu "num=1" -q gpu_a100 -o output10.log'

    cmd = f'{gpu_cmd} python {script_path} --folder_path {folder_path} -enum {experiment_number} -chnum {channel_number} -k {k} --l1_W {parameters.l1_W} -max_i {parameters.max_iter} --min_loss {parameters.min_loss} --batch_size {parameters.batch_size} --step_size {parameters.step_size} --min_diff {parameters.min_diff}'
    print(cmd)





