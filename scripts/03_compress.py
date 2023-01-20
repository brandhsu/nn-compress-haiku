"""Compress a Neural Network on CIFAR-10 using Haiku"""

import argparse
import json
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterator, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

Batch = Tuple[np.ndarray, np.ndarray]

sys.path.insert(0, "./")
from nn_compress_haiku import prune, quant, svd


def teacher_net_fn(batch: Batch) -> jnp.ndarray:
    """A simple convolutional feedforward deep neural network.

    Args:
        batch (Batch): A tuple containing (data, labels).

    Returns:
        jnp.ndarray: output of network
    """
    x = normalize(batch[0])

    net = hk.Sequential(
        [
            hk.Conv2D(output_channels=6 * 3, kernel_shape=(5, 5)),
            jax.nn.relu,
            hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="VALID"),
            jax.nn.relu,
            hk.Conv2D(output_channels=16 * 3, kernel_shape=(5, 5)),
            jax.nn.relu,
            hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="VALID"),
            hk.Flatten(),
            hk.Linear(3000),
            jax.nn.relu,
            hk.Linear(2000),
            jax.nn.relu,
            hk.Linear(2000),
            jax.nn.relu,
            hk.Linear(1000),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return net(x)


def student_net_fn(batch: Batch) -> jnp.ndarray:
    """A simple convolutional feedforward deep neural network.

    Args:
        batch (Batch): A tuple containing (data, labels).

    Returns:
        jnp.ndarray: output of network
    """
    x = normalize(batch[0])

    net = hk.Sequential(
        [
            hk.Conv2D(output_channels=6 * 3, kernel_shape=(5, 5)),
            jax.nn.relu,
            hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="VALID"),
            jax.nn.relu,
            hk.Conv2D(output_channels=16 * 3, kernel_shape=(5, 5)),
            jax.nn.relu,
            hk.AvgPool(window_shape=(2, 2), strides=(2, 2), padding="VALID"),
            hk.Flatten(),
            hk.Linear(10),
        ]
    )
    return net(x)


def compute_accuracy(net: hk.Module, params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the accuracy of the network

    Args:
        net (hk.Module): A module defining the model structure.
        params (hk.Params): A nested dict of model parameters.
        batch (Batch): A tuple containing (data, labels).

    Returns:
        jnp.ndarray: An accuracy value.
    """
    x, y = batch
    predictions = net.apply(params, batch)

    yhat = jnp.argmax(predictions, axis=-1)
    accuracy = jnp.mean(y == yhat)
    return accuracy


def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    seed: int,
) -> Iterator[tuple]:
    """Loads the dataset as a generator of batches.

    Args:
        split (str): The dataset split to use {train, test}.
        is_training (bool): Flag to allow for dataset shuffling.
        batch_size (int): The sampled batch size.

    Returns
        Iterator[tuple]: An iterable numpy array representing (batch[data], batch[labels]).
    """
    ds = tfds.load("cifar10", split=split, as_supervised=True).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=seed)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def normalize(images) -> jnp.ndarray:
    """Feature normalization based on mean and standard deviation.

    Args:
        images (numpy.ndarray): Arrays corresponding to an image, range [0, 255].

    Returns:
         jnp.ndarray: Images that have undergone normalization (standardization).
    """
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    mean = np.asarray(CIFAR10_MEAN)
    std = np.asarray(CIFAR10_STD)
    x = images.astype(jnp.float32) / 255.0
    x -= mean
    x /= std

    return x


def compress_params(params: hk.Params, compression_func, fraction: float) -> hk.Params:
    """Compresses model parameters.

    Args:
        params (hk.Params): A nested dict of model parameters.
        compression_func: Compression function, any of the functions defined in `nn_compress_haiku`.
        fraction (float): Compression fraction.

    Returns:
        hk.Params: A nested dict of compressed model parameters.
    """
    new_params = deepcopy(params)
    for layer in new_params.keys():
        if "conv" in layer:
            continue
        new_params[layer]["w"] = compression_func(new_params[layer]["w"], fraction)
    return new_params


def compute_eval_metrics(
    net: hk.Module, params: hk.Params, batch: Batch, n_samples: int
) -> Tuple[List[float], List[float]]:
    """Computes model accuracy and inference time.

    Args:
        net (hk.Module): A module defining the model structure.
        params (hk.Params): A nested dict of model parameters.
        batch (Batch): A tuple containing (data, labels).
        n_samples (int): Number of times to evaluate.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing n_samples of accuracies and latencies.
    """
    latency_list = []
    accuracy_list = []
    for _ in range(n_samples):
        start = time.time()
        acc = compute_accuracy(net, params, batch)
        end = time.time()
        latency = end - start
        latency_list.append(latency)
        accuracy_list.append(acc)

    return accuracy_list, latency_list


def eval_at_multiple_fractions(
    net: hk.Module,
    params: hk.Params,
    dataset: Iterator[tuple],
    compression_func,
    n_fractions: int = 10,
    n_samples: int = 50,
) -> Tuple[List[Tuple[float]], List[Tuple[float]]]:
    """Evaluate model accuracy and inference time at multiple compression fractions.

    Args:
        net (hk.Module): A module defining the model structure.
        params (hk.Params): A nested dict of model parameters.
        batch (Batch): A tuple containing (data, labels).
        compression_func: Compression function, any of the functions defined in `nn_compress_haiku`.
        n_fractions (int): Number of compression fractions (defaults to 10).
        n_samples (int): Number of times to evaluate (defaults to 50).

    Returns:
        Tuple[List[Tuple[float]], List[Tuple[float]]]: A tuple containing n_fractions of (fractions, accuracies) and (fractions, latencies).
    """
    fractions_and_accuracies = []
    fractions_and_latencies = []

    for fraction in np.linspace(0, 0.9, n_fractions):
        print(f"Evaluating the model at {fraction * 100:.2f}% compression")

        compression_fraction = fraction
        if "svd" in compression_func.__name__:
            compression_fraction = 1 - compression_fraction

        new_params = compress_params(params, compression_func, compression_fraction)
        accuracy, latency = compute_eval_metrics(
            net, new_params, next(dataset), n_samples
        )

        print(
            f"Compression Fraction / Accuracy: "
            f"{fraction:.2f} / {np.mean(accuracy):.3f}."
        )
        fractions_and_accuracies.append((fraction, np.mean(accuracy)))

        print(
            f"Compression Fraction / Latency: "
            f"{fraction:.2f} / {np.mean(latency):.4f}."
        )
        fractions_and_latencies.append((fraction, np.mean(latency)))

    return fractions_and_accuracies, fractions_and_latencies


def plot_accuracy(fractions_and_accuracies: List[Tuple[float]], title: str, path: str):
    fractions, accuracies = zip(*fractions_and_accuracies)
    plt.plot(fractions, accuracies)
    plt.title(f"{title.upper()}: Accuracy vs. Compression Fraction")
    plt.xlabel("Compression Fraction")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{path}/accuracy-{title}.png")
    plt.close()


def plot_latency(fractions_and_latencies: List[Tuple[float]], title: str, path: str):
    fractions, times = zip(*fractions_and_latencies)
    plt.plot(fractions, times)
    plt.title(f"{title.upper()}: Latency vs. Compression Fraction")
    plt.xlabel("Compression Fraction")
    plt.ylabel("Latency (sec)")
    plt.legend()
    plt.savefig(f"{path}/latency-{title}.png")
    plt.close()


def store_results(
    fractions_and_metrics: List[Tuple[float]], metric: str, title: str, path: str
):
    fractions, metrics = zip(*fractions_and_metrics)
    results = {"fraction": fractions, metric: [metric.item() for metric in metrics]}
    json.dump(results, open(f"{path}/{metric}-{title}.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress a Neural Network on CIFAR-10 using Haiku"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to a trained model",
    )
    parser.add_argument(
        "--compression-func",
        type=str,
        default="svd",
        choices=["prune", "quant", "svd"],
        help="Compression function (default: svd)",
    )
    parser.add_argument(
        "--n-fractions",
        type=int,
        default=10,
        help="Number of compression fractions to evaluate (default: 10)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of sample batches to evaluate (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Randomization seed (default: 42)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="figs",
        help="Directory to save plots (default: figs)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save results (default: logs)",
    )
    args = parser.parse_args()

    compression_func = {
        "prune": prune.prune,
        "quant": quant.quant,
        "svd": svd.svd,
    }

    # First, make the network.
    net_fn = (teacher_net_fn, student_net_fn)["-kd" in args.model_path]
    net = hk.without_apply_rng(hk.transform(net_fn))

    # Second, make test dataset.
    test = load_dataset(
        "test",
        is_training=False,
        batch_size=10000,
        seed=args.seed,
    )

    # Third, load the network.
    params = pickle.load(open(args.model_path, "rb"))

    # Fourth, evaluate the model at multiple compression fractions.
    fractions_and_accuracies, fractions_and_latencies = eval_at_multiple_fractions(
        net,
        params,
        test,
        compression_func[args.compression_func],
        args.n_fractions,
        args.n_samples,
    )

    # Fifth, save plots.
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    plot_accuracy(fractions_and_accuracies, args.compression_func, args.save_dir)
    plot_latency(fractions_and_latencies, args.compression_func, args.save_dir)

    # Sixth, save results.
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    store_results(
        fractions_and_accuracies, "accuracy", args.compression_func, args.log_dir
    )
    store_results(
        fractions_and_latencies, "latency", args.compression_func, args.log_dir
    )
