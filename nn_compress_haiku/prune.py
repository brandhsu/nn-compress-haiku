import jax.numpy as jnp


def prune(weight: jnp.ndarray, prune_fraction: float) -> jnp.ndarray:
    """Prunes low magnitude weights by replacing them with zeros.

    Args:
        weight (jnp.ndarray): weight matrix (m x n).
        prune_fraction (float): fraction of weights to prune.

    Returns:
        jnp.ndarray: sparsified weight matrix.
    """
    vector = weight.reshape(-1)
    num_zeros = round(prune_fraction * len(vector))
    idx = jnp.argsort(jnp.abs(vector))
    prune_idx = idx[:num_zeros]

    mask = jnp.ones_like(vector)
    mask = mask.at[prune_idx].set(0)

    return (vector * mask).reshape(weight.shape)
