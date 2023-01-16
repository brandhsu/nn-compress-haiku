import jax.numpy as jnp
from sklearn.cluster import KMeans


def quant(
    weight: jnp.ndarray, quant_fraction: float, random_state: int = 42
) -> jnp.ndarray:
    """Quantize weight matrix using K-means clustering

    Args:
        weight (jnp.ndarray): _description_
        quant_fraction (float): fraction of weights to quantize
        random_state (int): random seed for K-means clustering

    Returns:
        jnp.ndarray: quantized weight matrix
    """
    vect = jnp.reshape(weight.shape, -1)
    n_clusters = len(vect) - round(quant_fraction * len(vect))
    if n_clusters == 0:
        n_clusters += 1

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init="auto"
    ).fit(vect.reshape(-1, 1))
    centroids = jnp.array(
        [kmeans.cluster_centers_[label].item() for label in kmeans.labels_]
    )

    return jnp.reshape(centroids, weight.shape)
