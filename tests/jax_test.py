import jax.numpy as jnp
import jax

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
print("\n")


def sum_squared_error(x, y):
    return jnp.sum((x - y) ** 2)


def squared_error_with_aux(x, y):
    return sum_squared_error(x, y), x - y


dim = 3
x = jnp.ones(dim) * 2
y = jnp.ones(dim) * 10
print(x - y)
(val, aux), grad = jax.value_and_grad(fun=squared_error_with_aux, has_aux=True)(x, y)
print(val)
print(aux)
