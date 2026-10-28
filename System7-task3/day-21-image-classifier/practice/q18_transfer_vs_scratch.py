"""Q18 (Hard): Use MobileNetV2 transfer learning with a frozen base. Compare it with
a CNN trained from scratch, then visualize predictions.

Both models use the same training subset and number of epochs, so the comparison is fair.
Set TRAIN_SUBSET = None to use all 50,000 training images (slower).
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
from image_data import CLASS_NAMES, load_cifar10, to_unit_range  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
TRAIN_SUBSET = 10000
EPOCHS = 3
tf.keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = load_cifar10()
if TRAIN_SUBSET:
    x_train, y_train = x_train[:TRAIN_SUBSET], y_train[:TRAIN_SUBSET]
x_train, x_test = to_unit_range(x_train), to_unit_range(x_test)


def scratch_cnn():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def mobilenet_transfer():
    # MobileNetV2 was trained on ImageNet at 96×96 or larger, so we resize up.
    base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = False                       # feature extraction: keep pretrained weights fixed
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Resizing(96, 96)(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)   # 0..1 -> -1..1, what MobileNetV2 expects
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)  # our new 10-class head
    return tf.keras.Model(inputs, outputs)


results = {}
for name, builder in [("CNN from scratch", scratch_cnn), ("MobileNetV2 (frozen base)", mobilenet_transfer)]:
    model = builder()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=64, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    results[name] = (model, acc)
    print(f"{name:<28} test accuracy = {acc:.3f}   trainable params = {trainable:,}")

best_name = max(results, key=lambda k: results[k][1])
print(f"\nBetter on the test set: {best_name}")

# Visualize predictions: green title = correct, red = wrong
model = results[best_name][0]
idx = np.random.default_rng(0).choice(len(x_test), 9, replace=False)
probs = model.predict(x_test[idx], verbose=0)
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
for ax, i, p in zip(axes.ravel(), idx, probs):
    pred, true = int(p.argmax()), int(y_test[i])
    ax.imshow(x_test[i])
    ax.set_title(f"pred: {CLASS_NAMES[pred]}\ntrue: {CLASS_NAMES[true]}", fontsize=8,
                 color="green" if pred == true else "red")
    ax.axis("off")
fig.suptitle(f"Predictions – {best_name}")
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q18_predictions.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q18_predictions.png'}")
