"""Train a Neural Network using Knowledge Distillation on CIFAR-10 using Haiku"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

Batch = Tuple[np.ndarray, np.ndarray]

module = __import__("01_train", globals(), locals(), ["*"])
for k in dir(module):
    if not k.startswith("_"):
        locals()[k] = getattr(module, k)


def teacher_net_fn(batch: Batch) -> jnp.ndarray:
    # This references the net_fn in 01_train.py
    return net_fn(batch)


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


def compute_loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the loss of the student network against the outputs of the teacher including L2.

    Args:
        params (hk.Params): A nested dict of model parameters.
        batch (Batch): A tuple containing (data, labels, teacher logits).

    Returns:
        jnp.ndarray: A loss value.
    """
    x, y, targets = batch
    logits = student_net.apply(params, batch)
    labels = jax.nn.one_hot(y, 10)

    targets = jax.nn.softmax(targets)
    log_preds = jax.nn.log_softmax(logits)

    l2_loss = sum(optax.l2_loss(w).sum() for w in jax.tree_util.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * log_preds)
    kl_divergence = jnp.sum(targets * (jnp.log(targets) - log_preds))

    loss = alpha * softmax_xent + (1 - alpha) * kl_divergence

    return loss + (1e-4 * l2_loss)


def distill(batch: Batch) -> tuple:
    x, y = batch
    logits = teacher_net.apply(teacher_params, batch)
    return x, y, logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Neural Network using Knowledge Distillation on CIFAR-10 using Haiku"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to a trained teacher model",
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
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Loss weight between the teacher predictions and ground-truth labels "
            " given by the following formula: "
            " (alpha * ce(labels, student)) + (1 - alpha) * kl(teacher, student)"
            " (default: 0.0)"
        ),
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
    teacher_net = hk.without_apply_rng(hk.transform(teacher_net_fn))
    student_net = hk.without_apply_rng(hk.transform(student_net_fn))
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
    teacher_params = pickle.load(open(args.model_path, "rb"))
    student_params = student_avg_params = student_net.init(
        jax.random.PRNGKey(args.seed), next(train)
    )
    opt_state = opt.init(student_params)
    alpha = args.alpha

    # Fourth, training and evaluation loop.
    for step in range(args.train_steps):
        if step % args.eval_steps == 0:
            val_accuracy = compute_accuracy(
                student_net, student_avg_params, next(validation)
            )
            test_accuracy = compute_accuracy(
                student_net, student_avg_params, next(test)
            )
            val_accuracy, test_accuracy = jax.device_get((val_accuracy, test_accuracy))
            print(
                f"[Step {step}] Validation / Test accuracy: "
                f"{val_accuracy:.3f} / {test_accuracy:.3f}."
            )

        student_params, opt_state = update(
            student_params, opt_state, distill(next(train))
        )

        student_avg_params = ema_update(student_params, student_avg_params)

    # Fifth, save trained models.
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(student_params, open(f"{args.save_dir}/params_kd.pkl", "wb"))
    pickle.dump(student_avg_params, open(f"{args.save_dir}/avg_params_kd.pkl", "wb"))
