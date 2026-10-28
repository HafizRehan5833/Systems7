"""Q16 (Medium): Resize images to 64×64, normalize them, apply augmentation layers,
and display 9 augmented samples.

Augmentation layers only change images during training, so we call them with training=True.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "data"))
from image_data import CLASS_NAMES, load_cifar10  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
tf.keras.utils.set_random_seed(0)

(x_train, y_train), _ = load_cifar10()
images = tf.constant(x_train[:9])

preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(64, 64),              # 32×32 -> 64×64
    tf.keras.layers.Rescaling(1.0 / 255),          # normalize 0–255 -> 0–1
])
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
])

resized = preprocess(images)
augmented = augment(resized, training=True)   # random changes happen here

print(f"Original shape: {images.shape}, after resize: {resized.shape}")
print(f"Value range after normalization: {float(tf.reduce_min(resized)):.2f} to {float(tf.reduce_max(resized)):.2f}")

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for ax, img, label in zip(axes.ravel(), augmented.numpy(), y_train[:9]):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(CLASS_NAMES[label], fontsize=9)
    ax.axis("off")
fig.suptitle("9 augmented samples (64×64)")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q16_augmented_samples.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q16_augmented_samples.png'}")
