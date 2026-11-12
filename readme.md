# Macrograd

An autograd engine and neural network library in pure Python that works on N-dimensional tensors. It doesn't use numpy or even the `math` module. `exp` and `log` are Taylor series, with range reduction so they stay accurate for big and small inputs.

Like PyTorch, it records each op in a graph as it runs and backpropagates through that graph. It's slow on purpose. Every op is a plain Python loop over a flat list of numbers, so you can follow the whole forward and backward pass.

## Compared to micrograd

Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) is a great way to learn backprop, but it only works on scalars. Macrograd does the same thing on tensors. A tensor is a flat list with a shape and strides, so a whole layer runs as one `x @ W` instead of a loop over neurons. Layers hold a full weight matrix, like PyTorch's `nn.Linear`, and there's no `Neuron` class.

## Running it

The library only uses the standard library. `visualize.py`, and so `train.py`, also needs the `graphviz` package to draw graphs.

```sh
python3 test_macrograd.py   # tests
python3 train.py            # tiny MLP, then draws its graph
python3 train_iris.py       # MLP on iris
python3 train_digits.py     # CNN on handwritten digits
python3 train_gpt.py        # tiny GPT on Shakespeare
python3 train_lstm.py       # LSTM on the same text, to compare
```

The tests compare forward values with Python's `math` module and check gradients against finite differences. To check the backward pass of your own op, give `gradcheck` a function that returns one number:

```python
from engine import tensor, gradcheck

x = tensor([[0.5, -1.0], [2.0, 0.3]])
gradcheck(lambda a: (a.tanh() * a).sum(), x)  # True, or raises at the first mismatch
```

`train.py` fits a 3→4→4→1 MLP (relu, then leaky relu, with dropout) to four points with Adam for 100 epochs, prints the loss as it goes and renders the graph of the last loss to `macrograd_architecture.svg`.

`train_iris.py` trains a 4→16→3 classifier on the UCI iris dataset with mini-batches, AdamW and a cosine lr schedule, then saves the weights and loads them back. It takes about a second and gets 29 of 30 test flowers right (28 to 30 across random seeds 0 to 5).

`train_digits.py` trains a small CNN, two conv + max pool blocks and a linear layer, on UCI optdigits: 8x8 images of handwritten digits. The test digits were written by different people than the training digits. Five epochs take about a minute and get 97.0% of the 1797 test images right (96.4% to 97.0% across seeds 0 to 3).

`train_gpt.py` trains a tiny character-level GPT on tiny Shakespeare, 1.1 million characters from the plays. The model has token and position embeddings, one transformer block with 4 heads, 32 numbers per character and a 16-character context, 17,505 parameters in all. 1500 steps of 8 random windows take about 13 minutes. Guessing evenly among the 65 characters would score a loss of 4.17, and a bigram model that only looks at the previous character scores 2.48. The model gets 2.18 on held-out text (2.19 with another seed). Then it writes 300 characters after `ROMEO:`:

```text
ROMEO:
Whand seard fordter a ben fort is hear,
Wamon: thy your a knofile.

QUS:
Fo hand he sow, I stremay the con lay,
Selai wides the ome the des! I con that inift he of athe pron:
And speme, ank hothe the fou.
```

It has the layout of a play, with speaker names in capitals, colons and short lines, and some real words like "thy", "hand" and "that", but most of the words are made up.

`train_lstm.py` trains an LSTM on the same text, set up like the GPT: the same 16-character windows in the same order, the same held-out windows, optimizer, lr schedule and number of steps, and nearly the same size. It has 32 numbers per character, an `lstm_cell` with a hidden size of 42 and a linear layer, 17,475 parameters in all. 1500 steps take about 12 minutes.

| | GPT | LSTM |
|---|---|---|
| Parameters | 17,505 | 17,475 |
| Held-out loss after 500 steps | 2.38 | 2.23 |
| Held-out loss after 1500 steps | 2.18 (2.19 with another seed) | 2.12 (2.10 with another seed) |
| Writing 300 characters | 12.6 s | 0.6 s |

The LSTM is ahead the whole way. It writes text about 20 times faster because it reads each new character with one cell step and keeps the rest in its state, while the GPT runs its whole 16-character window through again for every character.

The GPT can't see further back than 16 characters, but the LSTM's state carries on for as long as you feed it. It was only trained on 16 characters at a time from a zero state, and a longer run still helps: if it first reads the 48 characters before each held-out window, it scores 2.06 on those windows, against 2.11 from a zero state.

This is a small setting, with 16 characters of context and about 17,500 parameters. Transformers took over from LSTMs at much bigger sizes and longer contexts, and on GPUs, where attention handles all the positions of a window in parallel and an LSTM has to go one step at a time.

The LSTM's sample has long lines and no speaker names:

```text
ROMEO:
ney the good he your:
Could is I the and head maes, we ton to, fore doter that to the are of the, shany upe ond my of with and mum a chest,
Fortmayt sher is afer the the and sting theres braip;
And cust, as we to thishat manou wast haves and ther'dter the I hest craints werbould thy the suce him.
```

The run with the other seed did write names and short lines, so one 300-character sample doesn't separate the two models. The held-out loss does.

The datasets download into `datasets/` on the first run, and the saved weights go in `checkpoints/`.

## What's in it

### engine.py

`tensor` holds a flat list of numbers with a shape and strides.

- Arithmetic: `+`, `-`, `*`, `/`, `**` and `@`. `@` also works on batches when the leading dims match.
- `+` and `*` broadcast like numpy. Shapes line up from the right and size-1 dims repeat, so `(3, 1) + (4,)` gives `(3, 4)`. `.expand(shape)` does it explicitly.
- Reductions: `.sum`, `.mean`, `.var`, `.max`, `.argmax`, `.min` and `.argmin` take `axis` and `keepdim` and reduce everything by default. There's also `.std()`.
- `.cumsum(axis)` keeps a running total along an axis, so `[1, 2, 3]` becomes `[1, 3, 6]`. The shape stays the same, and the last value along the axis is the sum.
- Shapes: `.reshape()` (one dim can be `-1`), `.transpose(dim0, dim1)` (the last two dims by default), `.flatten(start_dim)`, `concat(tensors, axis)` and `stack(tensors, axis)`, which joins same-shape tensors along a new axis. Indexing takes ints, slices and lists of ints: `x[0]`, `x[:, 1:3]`, `x[[2, 0]]`.
- Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu` (the tanh approximation), `silu` (`x * sigmoid(x)`, also called swish), and `softmax` and `log_softmax` over each row.
- `.clamp(min, max)` limits every value to a range. Either bound can be `None`, so `.clamp(min=0)` is relu. Values that got cut off get no gradient.
- `where(condition, a, b)` takes each value from `a` where the condition is true and from `b` elsewhere. All three broadcast, and `a` or `b` can be a number, like `where(mask, scores, float("-inf"))`. Each grad goes to the side that was picked.
- `.layer_norm(eps)` moves each row to mean 0 and variance 1, as one node with a handwritten backward.
- `.rms_norm(eps)` divides each row by its root mean square, `sqrt(mean(x^2) + eps)`. It's layer norm without subtracting the mean, as used in LLaMA.
- `.conv2d(w, stride, padding)` slides `(O, C, KH, KW)` filters over `(N, C, H, W)` images and zero-pads the edges. `.max_pool2d(kernel_size, stride)` keeps the largest value in each window of each channel, and `.avg_pool2d(kernel_size, stride)` takes the mean of each window. A window as big as the image gives each channel's mean.
- Losses: `mse_loss`, `cross_entropy_loss` (softmax per row, averaged over the batch, one node in the graph), `nll_loss(labels)` for integer labels after `log_softmax`, and `binary_cross_entropy_with_logits` for yes/no outputs, with the sigmoid built in.
- `accuracy(labels)` is the fraction of rows whose highest score is the right class.
- `gradcheck(f, *inputs)` checks any backward pass against finite differences.
- Inside `with no_grad():` ops don't record the graph, so nothing made in the block keeps its inputs alive. Use it when you only read `.data`, like when evaluating a model or generating text. A tensor made inside is a constant to any later `backward()`. `is_grad_enabled()` tells you whether ops are recording.

### nn.py

Layers subclass `module`. `.parameters()` finds every tensor a module stores, including ones inside sub-modules and lists, so a new layer only has to assign its tensors. `.train()` and `.eval()` switch dropout and batch norm between training and inference.

- `linear(nin, nout)` is a matmul plus a bias, with init scaled by fan-in. Inputs can have extra leading dims, like `(batch, time, features)`, and every row gets the same weights.
- `MLP(layer_sizes, activation_fns=None, dropout_p=None)` stacks linear layers. You can pass an activation function and a dropout rate for each hidden layer. By default every hidden layer uses relu and there's no dropout.
- `sequential(*layers)` runs layers in order. A layer can also be a plain function like `lambda x: x.relu()`.
- `dropout(p)` only drops values in training mode.
- `layer_norm(dim)` normalizes each example's features, then scales and shifts them with learned `gamma` and `beta`.
- `rms_norm(dim)` does the same with `.rms_norm()` and a learned `gamma`. It has no `beta`, since it never moves the mean.
- `batch_norm(dim)` normalizes each feature over the batch while training and keeps running averages of the mean and variance to use in `.eval()` mode. The running stats are buffers (see `buffer_names` and `.named_buffers()`), so they're saved with the weights, but the optimizer never changes them.
- `conv2d(in_channels, out_channels, kernel_size, stride, padding)` has square filters and one bias per filter. `max_pool2d(kernel_size)`, `avg_pool2d(kernel_size)` and `flatten()` wrap the tensor ops, and `flatten` keeps the batch dim, so a small CNN looks like `sequential(conv2d(1, 8, 3, padding=1), lambda x: x.relu(), max_pool2d(2), flatten(), linear(...))`.
- `embedding(num_embeddings, dim)` is a learned table with one vector per id. Ids of any shape come back as vectors of shape `ids.shape + (dim,)`.
- `sinusoidal_positions(length, dim)` is the fixed position table from the original transformer paper, to add to token vectors instead of a learned position embedding. Each pair of columns is a sin and a cos of the position, each pair at its own speed. Moving k positions on turns every pair by an angle that only depends on k, so the model can learn to look a fixed distance back, and the table works for any length.
- `rnn_cell(input_size, hidden_size)` is one step, `h = tanh(x @ w_x + h @ w_h + b)`. Loop it over a sequence and pass `h` along. Backprop through time comes from the graph.
- `lstm_cell(input_size, hidden_size)` is one LSTM step. Besides `h` it carries a cell state `c`, and learned gates decide how much of `c` to keep, what to write into it and how much of it to show in `h`. Call it as `h, c = cell(x, (h, c))`, or `cell(x)` to start from zeros. From one step to the next `c` is only multiplied by the forget gate and added to, so while that gate stays near 1 the gradient goes back through many steps without fading. The forget gate's bias starts at 1 for that reason.
- `gru_cell(input_size, hidden_size)` is one GRU step, a gated cell with no separate cell state: `h = cell(x, h)`. An update gate `z` mixes the old `h` with a new candidate `n`, `h = (1 - z) * n + z * h`, and a reset gate decides how much of the old `h` goes into `n`. It uses PyTorch's version of the formula, which applies the reset gate after multiplying `h` by its weights.
- `scaled_dot_product_attention(q, k, v, causal=False)` computes `softmax(q @ k^T / sqrt(d)) @ v` over any leading batch and head dims. With `causal=True`, position `i` only sees positions `0..i`.
- `multi_head_attention(dim, num_heads)` is self-attention on `(B, T, dim)` inputs. It projects to queries, keys and values, splits them into heads that attend separately, then joins the heads and projects again. Call it as `mha(x, causal=True)` when the model must not look ahead.
- `transformer_block(dim, num_heads, hidden=4 * dim)` is one GPT-style layer: `x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))`, where the MLP widens each position to `hidden` features with gelu. The `x +` residual paths give the gradient a direct way back to the input. The output has the same shape as the input, so blocks stack: `for block in blocks: x = block(x, causal=True)`.
- `sample(logits, temperature=1.0, top_k=None)` picks ids from a model's scores when generating text. It draws from `softmax(logits / temperature)`. A lower temperature favors the top scores (0 always takes the best one), a higher one spreads the picks out, and `top_k` only allows the k highest scores. Give it `(V,)` scores for one id or `(B, V)` for one per row.
- `xavier_uniform` and `xavier_normal` (for tanh and sigmoid), and `kaiming_uniform` and `kaiming_normal` (for relu), re-initialize a linear or conv2d weight in place, like `kaiming_normal(layer.w)`, so activations keep their size through deep networks.
- `.save(path)` writes the weights and buffers to a JSON file, and `.load(path)` reads them into a model with the same layout. Each tensor is stored under its `.named_parameters()` name, like `layers.0.w`.

### optim.py

- `SGD` with optional `momentum` and `weight_decay`.
- `Adagrad` gives each weight its own step size: it divides the step by the square root of the sum of all that weight's squared gradients so far. Weights with big gradients take small steps, and the steps keep shrinking as training goes on.
- `RMSprop` divides by a running average of recent squared gradients instead, so the steps stay about `lr` and don't shrink forever. With no correction for the average starting at 0, the first step is 10 times `lr` with the default `alpha=0.99`.
- `Adam`, which keeps a momentum and a velocity for each parameter. It's RMSprop plus momentum and that correction.
- `AdamW` is Adam with decoupled weight decay. It shrinks the weights directly instead of adding the decay to the gradient.
- `clip_grad_norm(parameters, max_norm)` scales all the gradients down together when their combined length is over `max_norm`. Call it between `backward()` and `step()`.
- `StepLR(optimizer, step_size, gamma)` multiplies the lr by `gamma` every `step_size` epochs. `CosineAnnealingLR(optimizer, T_max, eta_min)` lowers it from the starting lr to `eta_min` along a half cosine over `T_max` epochs. Call the schedule's `.step()` once per epoch, after `optimizer.step()`.
- `LinearWarmup(optimizer, warmup_steps, after=None)` starts at `lr / (warmup_steps + 1)` and raises the lr in equal steps to `lr` over the first `warmup_steps` steps. Then it hands over to `after` if you give it one, like `LinearWarmup(opt, 100, after=CosineAnnealingLR(opt, T_max=900))`. Small first steps give Adam's and RMSprop's gradient averages time to settle. It sets the small lr as soon as it's made, so the first step already uses it.

### data.py

`batches(x, y, batch_size, shuffle=True)` splits a dataset into mini-batches of rows, in a new random order each time. `y` can be a tensor or a list of integer labels.

### visualize.py

Every tensor keeps the tensors it came from (`_prev`) and the op that made it (`_op`). `draw_dot(loss)` walks back from `loss` and returns a Graphviz graph with an oval for each op and a box for each tensor, showing the first two values of its data and gradient.
