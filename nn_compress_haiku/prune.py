import jax.numpy as jnp


def prune(weight: jnp.ndarray, prune_fraction: float) -> jnp.ndarray:
    """Prunes low magnitude weights by replacing them with zeros.

    Args:
        weight (jnp.ndarray): weight matrix (m x n).
        prune_fraction (float): fraction of weights to prune.

    Returns:
        jnp.ndarray: sparsified weight matrix.
    """
    vect = weight.reshape(-1)
    num_zeros = round(prune_fraction * len(vect))
    idx = jnp.argsort(jnp.abs(vect))
    prune_idx = idx[:num_zeros]

    mask = jnp.ones_like(vect)
    mask = mask.at[prune_idx].set(0)

    return (vect * mask).reshape(weight.shape)
