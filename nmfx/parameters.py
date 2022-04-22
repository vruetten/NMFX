import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Parameters():
    def __init__(self, l1_W=0.0, max_iter=10000, min_loss=1e-3, batch_size=20, step_size=1e-2):
        self.l1_W = l1_W
        self.max_iter= max_iter
        self.min_loss = min_loss
        self.batch_size = batch_size
        self.step_size = step_size

    def tree_flatten(self):
        children = (self.l1_W, self.max_iter, self.min_loss, self.batch_size, self.step_size)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



class Log():
    def __init__(self,):
        self.l1_loss_W = []
        self.reconstruction_loss = []
        self.total_loss = []
