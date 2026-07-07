"""Train the EMNIST letters character recognition model."""

from __future__ import annotations

import os

import tensorflow as tf
import tensorflow_datasets as tfds

from model import (
    CHARACTER_MODEL_PATH,
    CHARACTER_NUM_CLASSES,
    SAVED_MODELS_DIR,
    build_character_model,
    compile_model,
)
from training_utils import evaluate_model, plot_training_history

NUM_EPOCHS = 25
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 3
LETTER_LABELS = [chr(ord("A") + i) for i in range(CHARACTER_NUM_CLASSES)]


def preprocess_character(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply EMNIST-specific preprocessing used in analysis/model_2.ipynb."""
    image = tf.cast(image, tf.float32)
    image = tf.transpose(image)
    image = image / 255.0
    image = tf.reshape(image, (28, 28, 1))
    label = label - 1
    return image, label


def prepare_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load EMNIST Letters and split into train, validation, and test sets."""
    emnist_dataset, emnist_info = tfds.load(
        "emnist/letters",
        with_info=True,
        as_supervised=True,
    )

    emnist_train = emnist_dataset["train"]
    emnist_test = emnist_dataset["test"]

    num_validation_samples = int(VALIDATION_SPLIT * emnist_info.splits["train"].num_examples)

    train_val_data = emnist_train.map(preprocess_character, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = emnist_test.map(preprocess_character, num_parallel_calls=tf.data.AUTOTUNE)

    validation_data = (
        train_val_data.take(num_validation_samples)
        .cache()
        .shuffle(10000, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    train_data = (
        train_val_data.skip(num_validation_samples)
        .cache()
        .shuffle(10000, seed=RANDOM_SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_data = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_data, validation_data, test_data


def train_character_model() -> None:
    """Train, evaluate, and save the character recognition model."""
    tf.random.set_seed(RANDOM_SEED)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    print("Loading EMNIST Letters dataset...")
    train_ds, val_ds, test_ds = prepare_datasets()

    print("Building character model...")
    model = compile_model(build_character_model())
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    print("Training character model...")
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

    evaluate_model(
        model,
        test_ds,
        target_names=LETTER_LABELS,
        report_title="Character Model",
    )

    model.save(CHARACTER_MODEL_PATH)
    print(f"\nModel saved to {CHARACTER_MODEL_PATH}")

    plot_path = os.path.join(SAVED_MODELS_DIR, "character_training_history.png")
    plot_training_history(history, plot_path, "Character Model")
    print(f"Training history saved to {plot_path}")


if __name__ == "__main__":
    train_character_model()
