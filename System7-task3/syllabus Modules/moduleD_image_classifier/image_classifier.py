"""Module D practical task: a CIFAR-10 image classifier with transfer learning."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from image_data import CLASS_NAMES, load_cifar10, to_unit_range  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "image_classifier.keras"
TRAIN_SUBSET = 10000     # use None for all 50,000 images
EPOCHS_HEAD = 3          # stage 1: train only the new head
EPOCHS_FINE = 2          # stage 2: fine-tune the top of the base
FINE_TUNE_LAYERS = 30    # how many top layers of MobileNetV2 to unfreeze
VAL_SIZE = 2000          # validation images taken from the END of the training set
tf.keras.utils.set_random_seed(42)


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Resizing(96, 96)(inputs)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)
    x = tf.keras.layers.RandomFlip("horizontal")(x)     # augmentation, active during training only
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def train(model, base, x_train, y_train, x_val, y_val):
    # Stage 1: feature extraction
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=EPOCHS_HEAD, batch_size=64, verbose=2)

    # Stage 2: fine-tuning with a much smaller learning rate, so pretrained
    # features are adjusted gently rather than overwritten
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=EPOCHS_FINE, batch_size=64, verbose=2)


def evaluate(model, x_test, y_test):
    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    print(f"\nTest accuracy: {(y_pred == y_test).mean():.3f}")
    print("Per-class accuracy:")
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test == i
        per_class[name] = (y_pred[mask] == i).mean()
        print(f"  {name:<11} {per_class[name]:.3f}")

    idx = np.random.default_rng(0).choice(len(x_test), 9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax, i in zip(axes.ravel(), idx):
        pred, true = y_pred[i], y_test[i]
        ax.imshow(x_test[i])
        ax.set_title(f"pred: {CLASS_NAMES[pred]}\ntrue: {CLASS_NAMES[true]}", fontsize=8,
                     color="green" if pred == true else "red")
        ax.axis("off")
    fig.suptitle("Prediction examples (green = correct)")
    plt.savefig(HERE / "predictions.png", dpi=120, bbox_inches="tight")


def predict_file(model, path: str) -> None:
    img = tf.keras.utils.load_img(path, target_size=(32, 32))
    arr = to_unit_range(tf.keras.utils.img_to_array(img)[np.newaxis])
    probs = model.predict(arr, verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    print(f"\nPredictions for {path}:")
    for i in top:
        print(f"  {CLASS_NAMES[i]:<11} {probs[i]:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 transfer-learning classifier")
    parser.add_argument("--predict", help="path to an image file to classify with the saved model")
    args = parser.parse_args()

    if args.predict:
        predict_file(tf.keras.models.load_model(MODEL_PATH), args.predict)
        return

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    if TRAIN_SUBSET:
        x_train, y_train = x_train[:TRAIN_SUBSET], y_train[:TRAIN_SUBSET]
    x_train, x_test = to_unit_range(x_train), to_unit_range(x_test)
    # Validation images come from the end of the training set; the test set is never used for tuning
    x_val, y_val = x_train[-VAL_SIZE:], y_train[-VAL_SIZE:]
    x_train, y_train = x_train[:-VAL_SIZE], y_train[:-VAL_SIZE]

    model, base = build_model()
    train(model, base, x_train, y_train, x_val, y_val)
    evaluate(model, x_test, y_test)
    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH.name}; predictions grid saved to predictions.png")


if __name__ == "__main__":
    main()
