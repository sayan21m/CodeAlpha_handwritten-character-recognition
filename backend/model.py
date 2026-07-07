"""CNN model architectures for digit and character recognition."""

import os

import tensorflow as tf

INPUT_SHAPE = (28, 28, 1)
DIGIT_NUM_CLASSES = 10
CHARACTER_NUM_CLASSES = 26

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
DIGIT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "digit_model.keras")
CHARACTER_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "character_model.keras")

# Legacy notebook filenames for backwards compatibility
LEGACY_DIGIT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "mnist_cnn.keras")
LEGACY_CHARACTER_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "emnist_cnn.keras")

MODEL_FALLBACKS = {
    DIGIT_MODEL_PATH: LEGACY_DIGIT_MODEL_PATH,
    CHARACTER_MODEL_PATH: LEGACY_CHARACTER_MODEL_PATH,
}

# Backwards-compatible aliases
MNIST_NUM_CLASSES = DIGIT_NUM_CLASSES
EMNIST_NUM_CLASSES = CHARACTER_NUM_CLASSES
NUM_CLASSES = DIGIT_NUM_CLASSES
MNIST_MODEL_PATH = DIGIT_MODEL_PATH
EMNIST_MODEL_PATH = CHARACTER_MODEL_PATH
MODEL_PATH = DIGIT_MODEL_PATH


def build_digit_model() -> tf.keras.Model:
    """Build the MNIST digit recognition CNN (analysis/model_1.ipynb)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(DIGIT_NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


def build_character_model() -> tf.keras.Model:
    """Build the EMNIST letters CNN (analysis/model_2.ipynb)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(CHARACTER_NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


build_model = build_digit_model
build_emnist_model = build_character_model


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile a model with the same settings used during training."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def resolve_model_path(path: str) -> str:
    """Resolve primary model path with legacy fallback filenames."""
    if os.path.exists(path):
        return path

    fallback = MODEL_FALLBACKS.get(path)
    if fallback and os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        f"Trained model not found at {path}. "
        f"Run train_digit.py / train_character.py or train.py first."
    )


def model_is_available(path: str) -> bool:
    """Check whether a model file exists at the primary or legacy path."""
    try:
        resolve_model_path(path)
        return True
    except FileNotFoundError:
        return False


def load_saved_model(path: str = DIGIT_MODEL_PATH) -> tf.keras.Model:
    """Load a trained Keras model from disk."""
    resolved_path = resolve_model_path(path)
    return tf.keras.models.load_model(resolved_path)
