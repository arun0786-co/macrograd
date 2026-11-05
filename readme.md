
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
* Reductions: `.sum(axis, keepdim)` (everything by default, or along one axis), `.mean(axis, keepdim)`, `.var(axis, keepdim)`, `.std()`, `.max(axis, keepdim)`, `.argmax(axis, keepdim)`.
* Shape ops: `.reshape()` (one dimension can be `-1` and is inferred), `.transpose(dim0, dim1)` (swaps the last two dims by default), indexing with ints, slices and lists of ints (`x[0]`, `x[:, 1:3]`, `x[[2, 0]]`). `concat(tensors, axis)` joins tensors along an axis.
* Broadcasting like numpy: `+` and `*` line shapes up from the right and repeat size-1 dims, so `(3, 1) + (4,)` gives `(3, 4)`. `.expand(shape)` does it directly.
* Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu` (tanh approximation), `softmax` and `log_softmax` (per row).
* Normalization: `.layer_norm(eps)` moves each row (last dim) to mean 0 and variance 1, as one node with its own backward.
* Loss: `mse_loss`, `cross_entropy_loss` (softmax per row, averaged over the batch, single-node backprop), `nll_loss(labels)` (integer class labels, use after `log_softmax`), `binary_cross_entropy_with_logits` (sigmoid built in, for yes/no outputs).
* `accuracy(labels)`: fraction of rows whose highest score is the right class.
* `gradcheck(f, *inputs)` checks any op's backward pass against finite differences.

**The Network (`nn`)**

* `module`, `linear`, `MLP`, `sequential`, `dropout`, `layer_norm`, `batch_norm` classes.
* `layer_norm(dim)` normalizes each example's features, then scales and shifts them with learned `gamma` and `beta`.
* `batch_norm(dim)` normalizes each feature over the batch in training and keeps running averages of the mean and variance for `.eval()` mode. The running stats are buffers (listed by `.named_buffers()`, set with `buffer_names`): saved with the weights but never touched by the optimizer.
* `.parameters()` finds every tensor a module stores, including inside sub-modules and lists, so a custom layer only has to store its tensors. `sequential(layer, ...)` chains modules and plain functions.
* Weights stored as matrices, so a whole layer is one matmul. Init is scaled by fan-in.
* Per-layer activation functions (pass callables) and dropout rates.
* Dropout layer with `.train()` and `.eval()` modes for training vs. inference.
* `.save(path)` writes the weights (and buffers) to a JSON file and `.load(path)` reads them back into a model of the same shape. Each tensor is stored under its name from `.named_parameters()`, like `layers.0.w`.

**The Optimizers (`optim`)**

* `SGD` with optional `momentum` and `weight_decay`.
* `Adam` with per-parameter momentum and velocity tracking.
* `AdamW`: Adam with decoupled weight decay (shrinks weights directly instead of adding to the gradient).
* `clip_grad_norm(parameters, max_norm)` scales all gradients down together when their combined length is over `max_norm`. Call it between `backward()` and `step()`.
* Learning rate schedules: `StepLR(optimizer, step_size, gamma)` multiplies the lr by `gamma` every `step_size` epochs, and `CosineAnnealingLR(optimizer, T_max, eta_min)` lowers it from the starting lr to `eta_min` along a half cosine over `T_max` epochs. Call `.step()` once per epoch, after `optimizer.step()`.

**Data (`data`)**

* `batches(x, y, batch_size, shuffle=True)` splits a dataset into mini-batches of rows, in a new random order each call. `y` can be a tensor or a list of integer labels.

## 🚀 Quick Start

See [train.py](train.py) for a full example (3→4→4→1 network, 100 epochs, loss tracking).

[train_iris.py](train_iris.py) trains a 4→16→3 classifier on the UCI iris dataset (downloaded on the first run) with mini-batches, AdamW and a cosine lr schedule, then saves the weights and loads them back. It runs in about a second and gets 29 of 30 test flowers right (28 to 30 across random seeds 0 to 5).

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
