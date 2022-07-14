import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Parameters():
    def __init__(self, l1_W=0.0, max_iter=10000, min_loss=1e-5, batch_size=20, step_size=1e-2, min_diff = 1e-7):
        self.l1_W = l1_W
        self.max_iter= max_iter
        self.min_loss = min_loss
        self.batch_size = batch_size
        self.step_size = step_size
        self.min_diff = min_diff

    def tree_flatten(self):
        children = (self.l1_W, self.max_iter, self.min_loss, self.batch_size, self.step_size, self.min_diff)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class IterationLog():

    def __init__(self, l1_loss_W=None, l1_loss_H=None, recon_loss=None,
            total_loss=None):

        self.l1_loss_W = l1_loss_W
        self.l1_loss_H = l1_loss_H
        self.reconstruction_loss = recon_loss
        self.total_loss = total_loss

    def tree_flatten(self):

        children = (self.l1_loss_W, self.l1_loss_H,
                    self.reconstruction_loss, self.total_loss)
        aux_data = None
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class FittingLog():

    def __init__(self,):

        self.l1_loss_W = []
        self.l1_loss_H = []
        self.reconstruction_loss = []
        self.total_loss = []
        self.grad_norm = []

