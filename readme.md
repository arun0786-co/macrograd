
# Macrograd

A lightweight, N-dimensional tensor autograd engine and neural network library built entirely from scratch—no numpy, no math library, just pure Python.

Macrograd implements a dynamic computational graph (DAG) for backpropagation, similar to PyTorch. Every op works on whole tensors (batched matmul, activations, losses), written as plain Python loops with no C backend.

## 🥊 Macrograd vs. Micrograd

If you know Andrej Karpathy's `micrograd`, you'll see why this exists. Micrograd is brilliant for learning, but it works on **scalars only**.

Macrograd goes further:

* **Tensors over Scalars:** Micrograd wraps individual numbers. Macrograd uses N-dimensional flat arrays with custom stride logic for real-world data.
* **Batched Matmul:** Micrograd loops over neurons individually. Macrograd does `X @ W` in a single pass—entire layers at once.
* **No External Math:** No `math` module, no numpy. Exponents and logs computed from Taylor Series approximations, with range reduction so they stay accurate for large and small inputs.
* **Layer-level modules:** each `linear` layer holds a full weight matrix (no `Neuron` abstraction), like PyTorch's `nn.Linear`.

## ⚙️ Features

**The Engine (`tensor`)**

* N-dimensional flat arrays with stride-based indexing.
* All ops: `+`, `-`, `*`, `/`, `**`, `@` (batched matmul).
* Reductions: `.sum(axis, keepdim)` (everything by default, or along one axis), `.mean()`, `.var()`, `.std()`.
* Shape ops: `.reshape()` (one dimension can be `-1` and is inferred), `.transpose(dim0, dim1)` (swaps the last two dims by default).
* Broadcasting like numpy: `+` and `*` line shapes up from the right and repeat size-1 dims, so `(3, 1) + (4,)` gives `(3, 4)`. `.expand(shape)` does it directly.
* Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu` (tanh approximation), `softmax` (per row).
* Loss: `mse_loss`, `cross_entropy_loss` (softmax per row, averaged over the batch, single-node backprop).
* `gradcheck(f, *inputs)` checks any op's backward pass against finite differences.

**The Network (`nn`)**

* `module`, `linear`, `MLP`, `dropout` classes.
* Weights stored as matrices, so a whole layer is one matmul. Init is scaled by fan-in.
* Per-layer activation functions (pass callables) and dropout rates.
* Dropout layer with `.train()` and `.eval()` modes for training vs. inference.

**The Optimizers (`optim`)**

* `SGD` with manual learning rate control.
* `Adam` with per-parameter momentum and velocity tracking.

## 🚀 Quick Start

See [train.py](train.py) for a full example (3→4→4→1 network, 100 epochs, loss tracking).

Run the tests with `python3 test_macrograd.py`. They compare forward values against Python's `math` module and check gradients against finite differences.

To check the gradients of your own op, pass a function that returns a single value:

```python
from engine import tensor, gradcheck

x = tensor([[0.5, -1.0], [2.0, 0.3]])
gradcheck(lambda a: (a.tanh() * a).sum(), x)  # True, or raises at the first mismatch
```

**Default behavior:**

* All hidden layers use `relu` if `activation_fns=None`
* No dropout if `dropout_p=None`

## 🧠 Graph Visualization

Every tensor tracks its `_prev` children and `_op` tag. Call `draw_dot(loss_tensor)` to export a Graphviz `.svg` showing the full computational graph—forward ops and backward gradients in one clean diagram.
