"""Q15 (Medium): Build, compile and train a Keras Sequential model on tabular data.
Plot loss curves and compare SGD with Adam.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info messages
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DATA = Path(__file__).resolve().parents[2] / "tasks" / "data" / "students.csv"
EPOCHS = 100
tf.keras.utils.set_random_seed(42)

df = pd.get_dummies(pd.read_csv(DATA), columns=["school_type"], drop_first=True).astype(float)
X = df.drop(columns="final_score").to_numpy()
y = df["final_score"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)            # fit on training data only
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)


def build_model(optimizer):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),                 # regression: one linear output
    ])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


results = {}
for name, optimizer in {
    "SGD (lr=0.01)": tf.keras.optimizers.SGD(learning_rate=0.01),
    "Adam (lr=0.001)": tf.keras.optimizers.Adam(learning_rate=0.001),
}.items():
    model = build_model(optimizer)
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS,
                        batch_size=32, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    results[name] = history.history
    print(f"{name:<18} test MSE = {test_loss:8.2f}   test MAE = {test_mae:6.2f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, hist in results.items():
    ax.plot(hist["loss"], label=f"{name} – train")
    ax.plot(hist["val_loss"], "--", label=f"{name} – validation")
ax.set(title="Loss curves: SGD vs Adam", xlabel="Epoch", ylabel="MSE loss", yscale="log")
ax.legend()
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "q15_sgd_vs_adam.png", dpi=120, bbox_inches="tight")
print(f"Saved plot to {FIG_DIR / 'q15_sgd_vs_adam.png'}")
