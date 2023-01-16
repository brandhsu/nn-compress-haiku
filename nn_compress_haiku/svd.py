import jax.numpy as jnp


def svd(weight: jnp.ndarray, rank_fraction: float) -> jnp.ndarray:
    """Computes the low rank approximation of a matrix using the SVD.

    Args:
        weight (jnp.ndarray): weight matrix (m x n).
        rank_fraction (float): fraction of rank to keep.

    Returns:
        jnp.ndarray: low rank approximation of weight matrix.
    """
    u, s, v = jnp.linalg.svd(weight, full_matrices=False)
    rank = round(rank_fraction * s.shape[-1])

    return u[:, :rank] * s[:rank] @ v[:rank, :]
