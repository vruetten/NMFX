import jax.numpy as jnp
import numpy as np


class Parameters():
    def __init__(self,):
        self.l1_W = 0.0
        self.max_iter= 10000
        self.min_loss = 1e-3
        self.batch_size = 20
        self.step_size = 1e-2


class Log():
    def __init__(self,):
        self.l1_loss_W = []
        self.reconstruction_loss = []
        self.total_loss = []
