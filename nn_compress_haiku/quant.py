import jax.numpy as jnp
from sklearn.cluster import MiniBatchKMeans


def quant(
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
    n_clusters = len(vect) - round(quant_fraction * len(vect))
    if n_clusters == 0:
        n_clusters += 1

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state).fit(
        vect.reshape(-1, 1)
    )
    centroids = jnp.array(
        [kmeans.cluster_centers_[label].item() for label in kmeans.labels_]
    )

    return centroids.reshape(weight.shape)
