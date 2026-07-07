"""Train the MNIST digit recognition model."""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

from model import DIGIT_MODEL_PATH, SAVED_MODELS_DIR, build_digit_model, compile_model
from training_utils import evaluate_model, plot_training_history

NUM_EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 3
DIGIT_LABELS = [str(i) for i in range(10)]


def preprocess_digit(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale MNIST images to [0, 1] and reshape for the CNN."""
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))
    return image, label


def prepare_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load MNIST and split into train, validation, and test sets."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    train_size = int(len(x_train) * (1 - VALIDATION_SPLIT))
    x_val = x_train[train_size:]
    y_val = y_train[train_size:]
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(preprocess_digit, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10000, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .map(preprocess_digit, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(preprocess_digit, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


def train_digit_model() -> None:
    """Train, evaluate, and save the digit recognition model."""
    tf.random.set_seed(RANDOM_SEED)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    print("Loading MNIST dataset...")
    train_ds, val_ds, test_ds = prepare_datasets()

    print("Building digit model...")
    model = compile_model(build_digit_model())
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    print("Training digit model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    evaluate_model(model, test_ds, target_names=DIGIT_LABELS, report_title="Digit Model")

    model.save(DIGIT_MODEL_PATH)
    print(f"\nModel saved to {DIGIT_MODEL_PATH}")

    plot_path = os.path.join(SAVED_MODELS_DIR, "digit_training_history.png")
    plot_training_history(history, plot_path, "Digit Model")
    print(f"Training history saved to {plot_path}")


if __name__ == "__main__":
    train_digit_model()
