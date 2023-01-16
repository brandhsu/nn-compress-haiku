import jax.numpy as jnp


def prune(weight: jnp.ndarray, prune_fraction: float) -> jnp.ndarray:
    """Prunes low magnitude weights by replacing them with zeros

    Args:
        weight (jnp.ndarray): weight matrix (m x n)
        prune_fraction (float): fraction of weights to prune

    Returns:
        jnp.ndarray: sparsified weight matrix
    """
    vect = jnp.reshape(weight.shape, -1)
    num_zeros = round(prune_fraction * len(vect))
    idx = jnp.argsort(jnp.abs(vect))
    prune_idx = idx[:num_zeros]

    mask = jnp.ones_like(vect)
    mask[prune_idx] = 0

    return jnp.reshape(vect * mask, weight.shape)
