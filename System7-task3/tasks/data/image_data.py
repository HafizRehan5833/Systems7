"""Helpers for the CIFAR-10 image dataset.

CIFAR-10 has 60,000 colour images (32x32 pixels) in 10 classes. Keras downloads
it automatically the first time (about 170 MB) and caches it on your machine.
"""
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_cifar10():
    """Return ((x_train, y_train), (x_test, y_test)); images are uint8 arrays of shape (n, 32, 32, 3)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train.ravel()), (x_test, y_test.ravel())


def to_unit_range(images: np.ndarray) -> np.ndarray:
    """Normalize pixel values from 0–255 to 0–1."""
    return images.astype("float32") / 255.0
