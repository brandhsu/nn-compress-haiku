import numpy as np
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
    vect = weight.reshape(-1)
    k = len(vect) - round(quant_fraction * len(vect))
    if k == 0:
        k += 1

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state).fit(
        vect.reshape(-1, 1)
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
    vect = weight.reshape(-1)
    k = len(vect) - round(quant_fraction * len(vect))
    if k == 0:
        k += 1

    samples = np.linspace(vect.min(), vect.max(), k)
    closest = lambda v: np.argmin(np.abs(v - samples))
    assign = np.vectorize(closest)

    return jnp.reshape(samples[assign(vect)], weight.shape)
