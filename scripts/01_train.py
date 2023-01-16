"""Train a Neural Network on CIFAR-10 using Haiku"""

import argparse
import pickle
from pathlib import Path
from typing import Iterator, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

Batch = Tuple[np.ndarray, np.ndarray]


def net_fn(batch: Batch) -> jnp.ndarray:
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
            hk.Linear(4000),
            jax.nn.relu,
            hk.Linear(3000),
            jax.nn.relu,
            hk.Linear(2000),
            jax.nn.relu,
            hk.Linear(1000),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return net(x)


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


def compute_loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the loss of the network, including L2.

    Args:
        params (hk.Params): A nested dict of model parameters.
        batch (Batch): A tuple containing (data, labels).

    Returns:
        jnp.ndarray: A loss value.
    """
    x, y = batch
    logits = net.apply(params, batch)
    labels = jax.nn.one_hot(y, 10)

    l2_loss = sum(optax.l2_loss(w).sum() for w in jax.tree_util.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))

    return softmax_xent + (1e-4 * l2_loss)


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


@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Batch,
) -> Tuple[hk.Params, optax.OptState]:
    """Computes one step of gradient descent.

    Args:
        params (hk.Params): A nested dict of model parameters.
        opt_state (optax.OptState): The state of the optimizer.
        batch (Batch): A tuple containing (data, labels).

    Returns:
        Tuple[hk.Params, optax.OptState]: Updated parameters and optimizer state.
    """
    grads = jax.grad(compute_loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state


@jax.jit
def ema_update(params, avg_params) -> hk.Params:
    """Incrementally update parameters via polyak averaging.

    Polyak averaging tracks an (exponential moving) average of the past parameters of a model, for use at test/evaluation time.

    Args:
        params (hk.Params): A nested dict of model parameters.
        avg_params (hk.Params): A nested dict of polyak averaged model parameters.

    Returns:
        hk.Params: Updated parameters based on polyak averaging.
    """
    return optax.incremental_update(params, avg_params, step_size=0.001)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Neural Network on CIFAR-10 using Haiku"
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=3001,
        help="Number of training steps (default: 3001)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N training steps (default: 100)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1000,
        help="Training batch size (default: 1000)",
    )
    parser.add_argument(
        "--valid-batch-size",
        type=int,
        default=10000,
        help="Validation batch size (default: 10000)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=10000,
        help="Test batch size (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate (default: 1e-3)",
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
        default="models",
        help="Directory to save models (default: models)",
    )
    args = parser.parse_args()

    # First, make the network and optimizer.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(args.lr)

    # Second, make datasets.
    train = load_dataset(
        "train[:80%]",
        is_training=True,
        batch_size=args.train_batch_size,
        seed=args.seed,
    )
    validation = load_dataset(
        "train[80%:]",
        is_training=False,
        batch_size=args.valid_batch_size,
        seed=args.seed,
    )
    test = load_dataset(
        "test",
        is_training=False,
        batch_size=args.test_batch_size,
        seed=args.seed,
    )

    # Third, initialize network and optimizer.
    params = avg_params = net.init(jax.random.PRNGKey(args.seed), next(train))
    opt_state = opt.init(params)

    # Fourth, training and evaluation loop.
    for step in range(args.train_steps):
        if step % args.eval_steps == 0:
            val_accuracy = compute_accuracy(net, avg_params, next(validation))
            test_accuracy = compute_accuracy(net, avg_params, next(test))
            val_accuracy, test_accuracy = jax.device_get((val_accuracy, test_accuracy))
            print(
                f"[Step {step}] Validation / Test accuracy: "
                f"{val_accuracy:.3f} / {test_accuracy:.3f}."
            )

        params, opt_state = update(params, opt_state, next(train))
        avg_params = ema_update(params, avg_params)

    # Fifth, save trained models.
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(params, open(f"{args.save_dir}/params.pkl", "wb"))
    pickle.dump(avg_params, open(f"{args.save_dir}/avg_params.pkl", "wb"))
