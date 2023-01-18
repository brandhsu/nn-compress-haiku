import jax.numpy as jnp
from sklearn.cluster import MiniBatchKMeans


# NOTE: This is very, very, very slow, normal kmeans would be a couple more verys...
def quant_kmeans(
    weight: jnp.ndarray, quant_fraction: float, random_state: int = 42
) -> jnp.ndarray:
    """Quantize weight matrix using Mini-Batch K-Means clustering (for performance reasons).

    Args:
        weight (jnp.ndarray): weight matrix (m x n).
        quant_fraction (float): fraction of weights to quantize.
        random_state (int): random seed for Mini-Batch K-Means clustering.

    Returns:
        jnp.ndarray: quantized weight matrix.
    """
    vector = weight.reshape(-1)
    unique = jnp.unique(vector).size
    k = unique - round(quant_fraction * unique)

    if k >= unique:
        return weight

    if k <= 0:
        return jnp.ones_like(weight) * (weight.min() + weight.max()) * 0.5

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state).fit(
        vector.reshape(-1, 1)
    )
    centroids = jnp.array(kmeans.cluster_centers_[kmeans.labels_])

    return centroids.reshape(weight.shape)


def quant(weight: jnp.ndarray, quant_fraction: float) -> jnp.ndarray:
    """Quantize weight matrix using linear quantization.

    Args:
        weight (jnp.ndarray): weight matrix (m x n).
        quant_fraction (float): fraction of weights to quantize.

    Returns:
        jnp.ndarray: quantized weight matrix.
    """
    vector = weight.reshape(-1)
    unique = jnp.unique(vector).size
    k = unique - round(quant_fraction * unique)

    if k >= unique:
        return weight

    if k <= 0:
        return jnp.ones_like(weight) * (weight.min() + weight.max()) * 0.5

    samples = jnp.linspace(vector.min(), vector.max(), k + 1)[1:-1]
    labels = argclosest(vector, samples)

    return samples[labels].reshape(weight.shape)


def argclosest(vector: jnp.ndarray, samples: jnp.ndarray) -> jnp.array:
    """Finds the index of the closest value in an array for each element in another array.

    Example:
        vector = jnp.array([1, 2, 3, 4, 5])
        samples = jnp.array([2, 7])
        indices = argclosest(vector, samples)
        assert (indices == jnp.array([0, 0, 0, 0, 1])).all()

        indices = argclosest(samples, vector)
        assert (indices == jnp.array([1, 4])).all()

    Args:
        vector (jnp.ndarray): input array.
        samples (jnp.ndarray): array being compared (must be sorted).

    Returns:
        jnp.array: indices of the closest
    """
    idx_r = jnp.searchsorted(samples, vector)
    idx_r = jnp.clip(idx_r, 0, samples.size - 1)
    idx_l = jnp.clip(idx_r - 1, 0, samples.size - 1)
    dist_r = samples[idx_r] - vector
    dist_l = vector - samples[idx_l]
    return jnp.where(dist_r <= dist_l, idx_r, idx_l)
