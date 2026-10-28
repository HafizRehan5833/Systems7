"""Q17 (Hard): Build a CNN with Conv2D and MaxPooling layers for CIFAR-10.
Report test accuracy and a confusion matrix.
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
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tasks" / "data"))
from image_data import CLASS_NAMES, load_cifar10, to_unit_range  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
EPOCHS = 10
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = load_cifar10()
x_train, x_test = to_unit_range(x_train), to_unit_range(x_test)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    # Each Conv2D slides learned 3×3 filters over the image to produce feature maps.
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),            # halves width and height: 32 -> 16
    tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),            # 16 -> 8
    tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),            # 8 -> 4
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),   # one probability per class
])
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=64, verbose=2)

probs = model.predict(x_test, verbose=0)
y_pred = probs.argmax(axis=1)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("CIFAR-10 CNN – confusion matrix")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q17_cnn_confusion_matrix.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q17_cnn_confusion_matrix.png'}")
