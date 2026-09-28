# ---------------------------------------------------------------------------
# SIMPLE NEURAL NETWORK ARCHITECTURE - BUILT FROM SCRATCH WITH NUMPY
# ---------------------------------------------------------------------------
# Week 6 deliverable: Neural Networks & Deep Learning Basics.
#
# Every Week 6 topic is implemented by hand (no PyTorch / TensorFlow), so the
# maths behind each piece is visible in the code:
#   1. Perceptron & layers        -> Perceptron class, Dense layer class
#   2. Activation functions       -> sigmoid, tanh, ReLU, Leaky ReLU (+ derivatives)
#   3. Backpropagation            -> Dense.backward + numerical gradient check
#   4. Loss functions & optimizers-> BCE / MSE loss, SGD / Momentum / Adam
#
# Runs end to end from a terminal:
#     python simple_neural_network.py
#
# Experiments performed:
#   A. Single perceptron on AND / OR / XOR  (shows why hidden layers are needed)
#   B. Multi-layer network solves XOR
#   C. Gradient check: backprop gradients vs. numerical gradients
#   D. Optimizer comparison (SGD vs Momentum vs Adam) on the "moons" dataset
#   E. Real data: Breast Cancer Wisconsin classification (30-16-8-1 network)
#
# Plots are written as PNG files into the plots/ folder and the trained
# weights of the final model are saved to breast_cancer_nn_weights.npz.
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, do not open a window

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# paths are resolved from this file, so the script works from any directory
BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
WEIGHTS_PATH = BASE_DIR / "breast_cancer_nn_weights.npz"

RANDOM_STATE = 42
EPS = 1e-12  # keeps log() and division away from zero


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# TOPIC 1a : THE PERCEPTRON
# ---------------------------------------------------------------------------
# A single neuron: output = step(w . x + b).
# Learning rule (Rosenblatt): w <- w + lr * (target - prediction) * x
# It can only learn LINEARLY SEPARABLE problems (AND, OR) - not XOR.
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = learning_rate

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

    def fit(self, X, y, epochs=20):
        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi[None, :])[0])
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            if errors == 0:  # converged: every sample classified correctly
                break
        return self


# ---------------------------------------------------------------------------
# TOPIC 2 : ACTIVATION FUNCTIONS (and their derivatives for backprop)
# ---------------------------------------------------------------------------
# Each derivative takes z (the pre-activation) and returns d(activation)/dz.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    return np.tanh(z)


def tanh_deriv(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def linear(z):
    return z


def linear_deriv(z):
    return np.ones_like(z)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "linear": (linear, linear_deriv),
}


# ---------------------------------------------------------------------------
# TOPIC 1b : A FULLY-CONNECTED (DENSE) LAYER
# ---------------------------------------------------------------------------
# Forward : Z = X W + b ,  A = activation(Z)
# Backward: given dL/dA from the next layer
#           dZ = dL/dA * activation'(Z)
#           dW = X^T dZ ,  db = sum(dZ) ,  dL/dX = dZ W^T  (sent to previous layer)
class Dense:
    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng or np.random.default_rng(RANDOM_STATE)
        self.activation_name = activation
        self.act, self.act_deriv = ACTIVATIONS[activation]
        # He init suits ReLU-family layers, Xavier init suits sigmoid / tanh
        if activation in ("relu", "leaky_relu"):
            scale = np.sqrt(2.0 / n_in)
        else:
            scale = np.sqrt(1.0 / n_in)
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros((1, n_out))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    @property
    def n_params(self):
        return self.W.size + self.b.size

    def forward(self, X):
        self.X = X                      # cached for the backward pass
        self.Z = X @ self.W + self.b
        return self.act(self.Z)

    def backward(self, dA):
        dZ = dA * self.act_deriv(self.Z)
        self.dW = self.X.T @ dZ
        self.db = dZ.sum(axis=0, keepdims=True)
        return dZ @ self.W.T


# ---------------------------------------------------------------------------
# TOPIC 4a : LOSS FUNCTIONS
# ---------------------------------------------------------------------------
# Each loss returns its value and the gradient dL/dy_pred (averaged over batch).
class BinaryCrossEntropy:
    """L = -mean( y log(p) + (1-y) log(1-p) ) - for binary classification."""
    name = "binary_cross_entropy"

    def loss(self, y_true, y_pred):
        p = np.clip(y_pred, EPS, 1 - EPS)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def gradient(self, y_true, y_pred):
        p = np.clip(y_pred, EPS, 1 - EPS)
        return (p - y_true) / (p * (1 - p)) / y_true.shape[0]


class MeanSquaredError:
    """L = mean( (y - p)^2 ) - the standard regression loss."""
    name = "mean_squared_error"

    def loss(self, y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    def gradient(self, y_true, y_pred):
        return 2.0 * (y_pred - y_true) / y_true.size


# ---------------------------------------------------------------------------
# TOPIC 4b : OPTIMIZERS
# ---------------------------------------------------------------------------
# All optimizers read layer.dW / layer.db (filled in by backprop) and update
# layer.W / layer.b in place.
class SGD:
    """Plain gradient descent: w <- w - lr * grad"""
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


class Momentum:
    """v <- beta * v + grad ;  w <- w - lr * v   (keeps rolling in a consistent direction)"""
    def __init__(self, learning_rate=0.1, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if i not in self.velocity:
                self.velocity[i] = [np.zeros_like(layer.W), np.zeros_like(layer.b)]
            vW, vb = self.velocity[i]
            vW[:] = self.beta * vW + layer.dW
            vb[:] = self.beta * vb + layer.db
            layer.W -= self.lr * vW
            layer.b -= self.lr * vb


class Adam:
    """Momentum (1st moment) + per-weight adaptive step size (2nd moment), bias-corrected."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.state = {}

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self.state:
                self.state[i] = {"mW": np.zeros_like(layer.W), "vW": np.zeros_like(layer.W),
                                 "mb": np.zeros_like(layer.b), "vb": np.zeros_like(layer.b)}
            s = self.state[i]
            for p, g, m, v in (("W", layer.dW, "mW", "vW"), ("b", layer.db, "mb", "vb")):
                s[m] = self.beta1 * s[m] + (1 - self.beta1) * g
                s[v] = self.beta2 * s[v] + (1 - self.beta2) * g ** 2
                m_hat = s[m] / (1 - self.beta1 ** self.t)
                v_hat = s[v] / (1 - self.beta2 ** self.t)
                setattr(layer, p, getattr(layer, p) - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))


# ---------------------------------------------------------------------------
# TOPIC 3 : THE NETWORK - FORWARD PASS, BACKPROPAGATION, TRAINING LOOP
# ---------------------------------------------------------------------------
class NeuralNetwork:
    """
    A stack of Dense layers.

    layer_sizes = [n_inputs, hidden_1, ..., n_outputs]
    hidden layers use `hidden_activation`, the last layer uses `output_activation`.
    """

    def __init__(self, layer_sizes, hidden_activation="relu",
                 output_activation="sigmoid", loss=None, seed=RANDOM_STATE):
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_output = i == len(layer_sizes) - 2
            act = output_activation if is_output else hidden_activation
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], act, rng))
        self.loss_fn = loss or BinaryCrossEntropy()
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ----- forward: input flows layer by layer to the output -----
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # ----- backward: the chain rule applied from the loss back to the input -----
    def backward(self, y_true, y_pred):
        grad = self.loss_fn.gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y, optimizer, epochs=100, batch_size=32,
            X_val=None, y_val=None, verbose_every=0, seed=RANDOM_STATE):
        y = y.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)
        rng = np.random.default_rng(seed)
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            # mini-batch gradient descent: shuffle, then update once per batch
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                y_pred = self.forward(X[idx])
                self.backward(y[idx], y_pred)
                optimizer.step(self.layers)

            self._record(X, y, "train")
            if X_val is not None:
                self._record(X_val, y_val, "val")

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                msg = (f"  epoch {epoch:4d} | loss {self.history['train_loss'][-1]:.4f}"
                       f" | acc {self.history['train_acc'][-1]:.3f}")
                if X_val is not None:
                    msg += (f" | val_loss {self.history['val_loss'][-1]:.4f}"
                            f" | val_acc {self.history['val_acc'][-1]:.3f}")
                print(msg)
        return self

    def _record(self, X, y, split):
        p = self.forward(X)
        self.history[f"{split}_loss"].append(self.loss_fn.loss(y, p))
        self.history[f"{split}_acc"].append(float(np.mean((p >= 0.5) == y)))

    def summary(self):
        print(f"  {'Layer':<10}{'Shape (in -> out)':<22}{'Activation':<14}{'Params':>8}")
        print("  " + "-" * 54)
        total = 0
        for i, layer in enumerate(self.layers, 1):
            name = "Output" if i == len(self.layers) else f"Hidden {i}"
            shape = f"{layer.W.shape[0]} -> {layer.W.shape[1]}"
            print(f"  {name:<10}{shape:<22}{layer.activation_name:<14}{layer.n_params:>8}")
            total += layer.n_params
        print("  " + "-" * 54)
        print(f"  {'Total trainable parameters':<46}{total:>8}")
        print(f"  Loss function: {self.loss_fn.name}")

    def save(self, path):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        np.savez(path, layer_sizes=np.array(self.layer_sizes), **arrays)


# ---------------------------------------------------------------------------
# GRADIENT CHECK - proves backprop is implemented correctly
# ---------------------------------------------------------------------------
# Numerical gradient: dL/dw ~ (L(w + h) - L(w - h)) / 2h
# If backprop is right, relative error between the two should be tiny (< 1e-6).
def gradient_check(net, X, y, h=1e-5):
    y = y.reshape(-1, 1)
    net.backward(y, net.forward(X))
    analytic = np.concatenate([np.r_[l.dW.ravel(), l.db.ravel()] for l in net.layers])

    numeric = []
    for layer in net.layers:
        for param in (layer.W, layer.b):
            it = np.nditer(param, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                original = param[idx]
                param[idx] = original + h
                loss_plus = net.loss_fn.loss(y, net.forward(X))
                param[idx] = original - h
                loss_minus = net.loss_fn.loss(y, net.forward(X))
                param[idx] = original
                numeric.append((loss_plus - loss_minus) / (2 * h))
    numeric = np.array(numeric)

    return np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))


# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> plots/{name}")


def plot_activation_functions():
    z = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, name in enumerate(["sigmoid", "tanh", "relu", "leaky_relu"]):
        fn, deriv = ACTIVATIONS[name]
        axes[0, col].plot(z, fn(z), color="#2a6fdb", lw=2)
        axes[0, col].set_title(name)
        axes[1, col].plot(z, deriv(z), color="#e07a1f", lw=2)
        axes[1, col].set_title(f"{name} derivative")
        for ax in axes[:, col]:
            ax.axhline(0, color="grey", lw=0.6)
            ax.axvline(0, color="grey", lw=0.6)
            ax.grid(alpha=0.3)
    fig.suptitle("Activation functions (top) and their derivatives used in backprop (bottom)")
    save_fig(fig, "01_activation_functions.png")


def plot_architecture(layer_sizes, name, title, max_nodes=10):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_layers = len(layer_sizes)
    positions = []
    for i, size in enumerate(layer_sizes):
        shown = min(size, max_nodes)
        ys = np.linspace(0, 1, shown + 2)[1:-1]
        positions.append([(i, y) for y in ys])
        label = "Input" if i == 0 else ("Output" if i == n_layers - 1 else f"Hidden {i}")
        ax.text(i, -0.08, f"{label}\n({size} units)", ha="center", va="top", fontsize=10)
        if size > max_nodes:
            ax.text(i, 1.02, f"showing {max_nodes} of {size}", ha="center", fontsize=8, color="grey")
    for left, right in zip(positions[:-1], positions[1:]):
        for (x1, y1) in left:
            for (x2, y2) in right:
                ax.plot([x1, x2], [y1, y2], color="grey", lw=0.4, alpha=0.5)
    colors = ["#2a6fdb"] + ["#3fa34d"] * (n_layers - 2) + ["#e07a1f"]
    for pts, c in zip(positions, colors):
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=400, color=c, edgecolor="black", zorder=3)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.25, 1.08)
    ax.axis("off")
    ax.set_title(title)
    save_fig(fig, name)


def plot_decision_boundary(net, X, y, name, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    probs = net.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="black", s=25)
    ax.set_title(title)
    save_fig(fig, name)


# ---------------------------------------------------------------------------
# EXPERIMENT A : PERCEPTRON ON LOGIC GATES
# ---------------------------------------------------------------------------
def experiment_perceptron():
    banner("EXPERIMENT A : SINGLE PERCEPTRON ON LOGIC GATES")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    gates = {"AND": np.array([0, 0, 0, 1]),
             "OR": np.array([0, 1, 1, 1]),
             "XOR": np.array([0, 1, 1, 0])}
    for gate, y in gates.items():
        p = Perceptron(n_inputs=2).fit(X, y, epochs=50)
        acc = np.mean(p.predict(X) == y)
        status = "learned" if acc == 1.0 else "FAILED (not linearly separable)"
        print(f"  {gate:<4} predictions {p.predict(X)}  target {y}  accuracy {acc:.2f}  -> {status}")
    print("\n  A single perceptron draws one straight line, so XOR is impossible for it.")
    print("  Adding a hidden layer fixes this (Experiment B).")
    return X, gates["XOR"]


# ---------------------------------------------------------------------------
# EXPERIMENT B : MULTI-LAYER NETWORK SOLVES XOR
# ---------------------------------------------------------------------------
def experiment_xor(X, y):
    banner("EXPERIMENT B : 2-4-1 NEURAL NETWORK SOLVES XOR")
    net = NeuralNetwork([2, 4, 1], hidden_activation="tanh", output_activation="sigmoid")
    net.summary()
    net.fit(X, y, Adam(learning_rate=0.05), epochs=500, batch_size=4)
    probs = net.predict_proba(X).ravel()
    for xi, target, p in zip(X.astype(int), y, probs):
        print(f"  input {xi}  target {target}  predicted prob {p:.3f}  -> class {int(p >= 0.5)}")
    print(f"  XOR accuracy: {np.mean((probs >= 0.5) == y):.2f}")


# ---------------------------------------------------------------------------
# EXPERIMENT C : GRADIENT CHECK
# ---------------------------------------------------------------------------
def experiment_gradient_check():
    banner("EXPERIMENT C : BACKPROPAGATION GRADIENT CHECK")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = (rng.random(10) > 0.5).astype(float)
    for act in ["sigmoid", "tanh", "leaky_relu"]:
        net = NeuralNetwork([3, 5, 4, 1], hidden_activation=act, seed=1)
        err = gradient_check(net, X, y)
        verdict = "PASS" if err < 1e-6 else "CHECK"
        print(f"  hidden activation {act:<11} relative error {err:.2e}  -> {verdict}")
    print("  (ReLU is skipped: its kink at 0 makes numerical gradients unreliable.)")


# ---------------------------------------------------------------------------
# EXPERIMENT D : OPTIMIZER COMPARISON ON THE MOONS DATASET
# ---------------------------------------------------------------------------
def experiment_optimizers():
    banner("EXPERIMENT D : SGD vs MOMENTUM vs ADAM (moons dataset, 2-16-16-1)")
    X, y = make_moons(n_samples=600, noise=0.2, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    optimizers = {"SGD (lr=0.1)": lambda: SGD(0.1),
                  "Momentum (lr=0.1)": lambda: Momentum(0.1, beta=0.9),
                  "Adam (lr=0.01)": lambda: Adam(0.01)}
    fig, ax = plt.subplots(figsize=(8, 5))
    best_net, best_acc = None, -1.0
    for label, make_opt in optimizers.items():
        net = NeuralNetwork([2, 16, 16, 1], hidden_activation="relu", seed=RANDOM_STATE)
        net.fit(X_train, y_train, make_opt(), epochs=150, batch_size=32)
        acc = accuracy_score(y_test, net.predict(X_test).ravel())
        print(f"  {label:<18} final train loss {net.history['train_loss'][-1]:.4f}"
              f"  test accuracy {acc:.3f}")
        ax.plot(net.history["train_loss"], label=label, lw=2)
        if acc > best_acc:
            best_net, best_acc = net, acc
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy loss")
    ax.set_title("Training loss by optimizer (same network, same initial weights)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "02_optimizer_comparison.png")
    plot_decision_boundary(best_net, X, y, "03_moons_decision_boundary.png",
                           f"Learned non-linear decision boundary (test acc {best_acc:.3f})")


# ---------------------------------------------------------------------------
# EXPERIMENT E : REAL DATASET - BREAST CANCER CLASSIFICATION
# ---------------------------------------------------------------------------
def experiment_breast_cancer():
    banner("EXPERIMENT E : BREAST CANCER CLASSIFICATION (real dataset)")
    data = load_breast_cancer()
    X, y = data.data, data.target  # target: 1 = benign, 0 = malignant
    print(f"  samples: {X.shape[0]}   features: {X.shape[1]}   "
          f"benign: {int(y.sum())}   malignant: {int((1 - y).sum())}")

    # 60 / 20 / 20 train / validation / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

    # neural networks need scaled inputs; fit the scaler on training data only
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

    layer_sizes = [X.shape[1], 16, 8, 1]
    net = NeuralNetwork(layer_sizes, hidden_activation="relu", output_activation="sigmoid")
    print("\n  Network architecture:")
    net.summary()
    plot_architecture(layer_sizes, "04_network_architecture.png",
                      "Breast cancer classifier: 30 -> 16 (ReLU) -> 8 (ReLU) -> 1 (sigmoid)")

    print("\n  Training with Adam (lr=0.005, batch 32, 100 epochs):")
    net.fit(X_train, y_train, Adam(learning_rate=0.005), epochs=100, batch_size=32,
            X_val=X_val, y_val=y_val, verbose_every=20)

    y_pred = net.predict(X_test).ravel()
    print("\n  Test set results:")
    print(f"    accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"    precision: {precision_score(y_test, y_pred):.3f}")
    print(f"    recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"    F1 score : {f1_score(y_test, y_pred):.3f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    confusion matrix [rows = actual malignant/benign]:\n{cm}")

    # learning curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, metric, ylabel in ((axes[0], "loss", "BCE loss"), (axes[1], "acc", "accuracy")):
        ax.plot(net.history[f"train_{metric}"], label="train", lw=2)
        ax.plot(net.history[f"val_{metric}"], label="validation", lw=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning curves - breast cancer network")
    save_fig(fig, "05_learning_curves.png")

    # confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["malignant", "benign"])
    ax.set_yticks([0, 1], ["malignant", "benign"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (test set)")
    save_fig(fig, "06_confusion_matrix.png")

    net.save(WEIGHTS_PATH)
    print(f"\n  trained weights saved -> {WEIGHTS_PATH.name}")


def main():
    banner("WEEK 6 : SIMPLE NEURAL NETWORK ARCHITECTURE (NumPy, from scratch)")
    plot_activation_functions()
    X_xor, y_xor = experiment_perceptron()
    experiment_xor(X_xor, y_xor)
    experiment_gradient_check()
    experiment_optimizers()
    experiment_breast_cancer()
    banner("DONE - see the plots/ folder for all figures")


if __name__ == "__main__":
    main()
