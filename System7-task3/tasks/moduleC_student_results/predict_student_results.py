"""Module C practical task: predict final student scores with a neural network."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "students.csv"
TARGET = "final_score"
tf.keras.utils.set_random_seed(42)


def load_splits():
    df = pd.get_dummies(pd.read_csv(DATA), columns=["school_type"], drop_first=True).astype(float)
    X = df.drop(columns=TARGET).to_numpy()
    y = df[TARGET].to_numpy()
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_train)  # training data only
    return (scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test),
            y_train, y_val, y_test)


def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:<22} MAE = {mean_absolute_error(y_true, y_pred):6.2f}   "
          f"RMSE = {rmse:6.2f}   R² = {r2_score(y_true, y_pred):.3f}")


def build_network(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),              # randomly switch off 20% of units to reduce overfitting
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),                  # linear output for a numeric prediction
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    print("=== Test-set comparison ===")
    baseline = LinearRegression().fit(X_train, y_train)
    report("Linear regression", y_test, baseline.predict(X_test))

    model = build_network(X_train.shape[1])
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=300, batch_size=32, callbacks=[early], verbose=0)
    epochs_run = len(history.history["loss"])
    nn_pred = model.predict(X_test, verbose=0).ravel()
    report("Neural network", y_test, nn_pred)
    print(f"Epochs trained before early stopping: {epochs_run}")

    # Learning curve: a healthy run shows train and validation loss falling together
    plt.figure(figsize=(7, 4.5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Learning curve – student results network")
    plt.legend()
    plt.savefig(HERE / "learning_curve.png", dpi=120, bbox_inches="tight")

    # Where does the network miss by the most?
    errors = np.abs(nn_pred - y_test)
    print(f"\nTest predictions off by more than 10 points: {(errors > 10).sum()} of {len(errors)}")
    worst = np.argsort(errors)[-5:][::-1]
    print("Five largest errors (actual, predicted):")
    for i in worst:
        print(f"  actual {y_test[i]:6.1f}   predicted {nn_pred[i]:6.1f}")


if __name__ == "__main__":
    main()
