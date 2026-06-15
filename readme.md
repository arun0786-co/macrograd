
# Macrograd

A lightweight, N-dimensional tensor autograd engine and neural network library built entirely from scratch—no numpy, no math library, just pure Python.

Macrograd implements a dynamic computational graph (DAG) for backpropagation, similar to PyTorch. Fully vectorized to handle batched matrix ops and complex activations without external C backends.

## 🥊 Macrograd vs. Micrograd

If you know Andrej Karpathy's `micrograd`, you'll see why this exists. Micrograd is brilliant for learning, but it works on **scalars only**.

Macrograd goes further:

* **Tensors over Scalars:** Micrograd wraps individual numbers. Macrograd uses N-dimensional flat arrays with custom stride logic for real-world data.
* **Batched Matmul:** Micrograd loops over neurons individually. Macrograd does `X @ W` in a single pass—entire layers at once.
* **No External Math:** No `math` module, no numpy. Exponents and logs computed from Taylor Series approximations.
* **Production Layers:** Direct `Linear` transformations (no `Neuron` abstraction), scales like PyTorch.

## ⚙️ Features

**The Engine (`tensor`)**

* N-dimensional flat arrays with stride-based indexing.
* All ops: `+`, `-`, `*`, `/`, `**`, `@` (batched matmul).
* Reductions: `.sum()`, `.mean()`, `.var()`, `.std()`.
* Broadcasting via `.expand()` for scalars and bias rows.
* Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`.
* Loss: `mse_loss`, `cross_entropy_loss` (single-node backprop).

**The Network (`nn`)**

* `Module`, `Linear`, `MLP` classes.
* All layers vectorized; weights stored as matrices.
* Per-layer activation functions (pass callables) and dropout rates.
* Dropout layer with `.train()` and `.eval()` modes for training vs. inference.

**The Optimizers (`optim`)**

* `SGD` with manual learning rate control.
* `Adam` with per-parameter momentum and velocity tracking.

## 🚀 Quick Start

See [train.py](train.py) for a full example (3→4→4→1 network, 100 epochs, loss tracking).

### Per-Layer Activations and Dropout

```python
# Define activations for each hidden layer (last layer has None)
activations = [
    lambda x: x.relu(),
    lambda x: x.leaky_relu(alpha=0.01)
]

# Define dropout rates for each hidden layer
dropout_rates = [0.2, 0.1]

# Create model with custom activation functions and dropout per layer
model = MLP([3, 4, 4, 1], activation_fns=activations, dropout_p=dropout_rates)

# Training mode (dropout enabled)
model.train()
for epoch in range(epochs):
    y_pred = model(X)
    loss = y_pred.mse_loss(Y)
    # ... backward & optimizer step

# Evaluation mode (dropout disabled)
model.eval()
y_pred = model(X)
```

**Default behavior:**
- All hidden layers use `relu` if `activation_fns=None`
- No dropout if `dropout_p=None`

## 🧠 Graph Visualization

Every tensor tracks its `_prev` children and `_op` tag. Call `draw_dot(loss_tensor)` to export a Graphviz `.svg` showing the full computational graph—forward ops and backward gradients in one clean diagram.
