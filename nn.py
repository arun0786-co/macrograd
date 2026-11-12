import json
import random
try:
    from .engine import tensor
    from .optim import PI, _cos
except ImportError:
    # run as a script from this folder, not imported as a package
    from engine import tensor
    from optim import PI, _cos

class module:
    def __init__(self):
        self.training = True

    def zero_grad(self):
        for p in self.parameters():
            for i in range(len(p.data)):
                p.grad[i] = 0.0

    def children(self):
        # sub-modules stored as attributes, or inside a list or tuple attribute,
        # in the order they were assigned
        for v in vars(self).values():
            for item in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(item, module):
                    yield item

    # names of tensor attributes that hold running state instead of weights, like
    # batch_norm's running mean. parameters() leaves them out, so the optimizer never
    # changes them, but save() and load() include them
    buffer_names = ()

    def _named_tensors(self, buffers):
        # every tensor stored as an attribute (or in a list), plus those of every sub-module,
        # so a new layer only has to store its tensors. each one is named by where it is
        # stored: 'layers.0.w' is the w of the first module in self.layers. a tensor shared
        # by two layers is listed once, under the first name found, so the optimizer doesn't
        # update it twice per step. buffers picks the buffers instead of the parameters
        found_all, seen = [], set()
        for attr, v in vars(self).items():
            if isinstance(v, (list, tuple)):
                items = [(f"{attr}.{i}", item) for i, item in enumerate(v)]
            else:
                items = [(attr, v)]
            for name, item in items:
                if isinstance(item, tensor):
                    if (attr in self.buffer_names) != buffers:
                        continue
                    found = [(name, item)]
                elif isinstance(item, module):
                    found = [(f"{name}.{sub}", t) for sub, t in item._named_tensors(buffers)]
                else:
                    continue
                for full_name, t in found:
                    if id(t) not in seen:
                        seen.add(id(t))
                        found_all.append((full_name, t))
        return found_all

    def named_parameters(self):
        return self._named_tensors(buffers=False)

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def named_buffers(self):
        return self._named_tensors(buffers=True)

    def save(self, path):
        # writes every parameter and buffer to a JSON file as {name: {"shape": [...], "data": [...]}}.
        # json writes floats with enough digits to read back the exact same number
        tensors = self.named_parameters() + self.named_buffers()
        state = {name: {"shape": list(t.shape), "data": t.data} for name, t in tensors}
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path):
        # reads a file written by save() into this module's parameters and buffers. every
        # name and shape must match, so loading weights into a different model fails loudly.
        # the values are copied into the existing tensors, so an optimizer holding them keeps working
        with open(path) as f:
            state = json.load(f)
        tensors = dict(self.named_parameters() + self.named_buffers())
        missing = [name for name in tensors if name not in state]
        unexpected = [name for name in state if name not in tensors]
        if missing or unexpected:
            raise ValueError(f"Weights don't match the model: missing {missing}, unexpected {unexpected}")
        for name, t in tensors.items():
            shape = tuple(state[name]["shape"])
            if shape != t.shape or len(state[name]["data"]) != len(t.data):
                raise ValueError(f"{name} has shape {shape} in the file but {t.shape} in the model")
        for name, t in tensors.items():
            t.data[:] = [float(v) for v in state[name]["data"]]
    
    def train(self):
        self.training = True
        for child in self.children():
            child.train()
    
    def eval(self):
        self.training = False
        for child in self.children():
            child.eval()
    
'''class neuron(module):
    def __init__(self, nin):
        w_data = [random.uniform(-1,1) for _ in range(nin)]
        self.w = tensor(w_data)

        self.b = tensor([0.0])

    def __call__(self,x):
        act = self.w.dot(x) + self.b
        out = act.relu()
        return out
    
    def parameters(self):
        return [self.w, self.b] '''
    
class linear(module):
    def __init__(self, nin, nout):
        # nout no of op features ,
        # nin no of  ip features ,
        super().__init__()
        # scale by fan-in so wider layers don't blow up activations (same bound as pytorch's default)
        bound = nin ** -0.5
        w_data = [[random.uniform(-bound, bound) for _ in range(nout)] for _ in range(nin)]
        self.w = tensor(w_data)

        b_data = [[0.0 for _ in range(nout)]]
        self.b = tensor(b_data)

    def __call__(self,x):
        if len(x.shape) == 2:
            return x.matmul(self.w) + self.b
        # any other shape, like (batch, time, features) or a single (features,) row:
        # the leading dims are flattened into rows for the matmul and put back after
        lead = x.shape[:-1]
        out = x.reshape(-1, x.shape[-1]).matmul(self.w) + self.b
        return out.reshape(lead + (self.w.shape[1],))
    
    def __repr__(self):
        return f"Linear(in_features={self.w.shape[0]}, out_features={self.w.shape[1]})"


# weight init. each output of x @ w adds up fan_in terms w * x, so its variance is
# fan_in * var(w) * E[x^2]. picking var(w) to cancel fan_in keeps activations the same
# size from layer to layer instead of growing or shrinking with depth.
# these fill a weight in place and return it, e.g. kaiming_normal(layer.w)

def _fans(w):
    # a linear weight is (inputs, outputs). a conv2d weight is (O, C, KH, KW): each output
    # adds up C * KH * KW inputs, and each input reaches O * KH * KW outputs
    if len(w.shape) == 2:
        return w.shape[0], w.shape[1]
    if len(w.shape) == 4:
        O, C, KH, KW = w.shape
        return C * KH * KW, O * KH * KW
    raise ValueError(f"init expects a linear (fan_in, fan_out) or conv2d (O, C, KH, KW) weight, got shape {w.shape}")


def _fill(w, std, normal):
    # mean 0 values with the given std. uniform(-a, a) has variance a^2 / 3, so a = sqrt(3) * std
    if normal:
        w.data[:] = [random.gauss(0.0, std) for _ in w.data]
    else:
        bound = 3 ** 0.5 * std
        w.data[:] = [random.uniform(-bound, bound) for _ in w.data]
    return w


def xavier_uniform(w, gain=1.0):
    # Glorot and Bengio (2010). forward, activations keep their size with var(w) = 1 / fan_in;
    # backward, gradients keep theirs with 1 / fan_out. xavier takes the average of the two:
    # var(w) = gain^2 * 2 / (fan_in + fan_out). meant for tanh and sigmoid, which are close
    # to linear near 0
    fan_in, fan_out = _fans(w)
    return _fill(w, gain * (2.0 / (fan_in + fan_out)) ** 0.5, normal=False)


def xavier_normal(w, gain=1.0):
    fan_in, fan_out = _fans(w)
    return _fill(w, gain * (2.0 / (fan_in + fan_out)) ** 0.5, normal=True)


def kaiming_uniform(w, negative_slope=0.0):
    # He et al. (2015), for relu. relu zeroes the negative half of its inputs, so
    # E[relu(z)^2] = var(z) / 2, and var(w) = 2 / fan_in makes up for that lost half.
    # for leaky_relu with slope a, the negative half keeps a^2 of its size:
    # var(w) = 2 / ((1 + a^2) * fan_in)
    fan_in, _ = _fans(w)
    return _fill(w, (2.0 / ((1 + negative_slope ** 2) * fan_in)) ** 0.5, normal=False)


def kaiming_normal(w, negative_slope=0.0):
    fan_in, _ = _fans(w)
    return _fill(w, (2.0 / ((1 + negative_slope ** 2) * fan_in)) ** 0.5, normal=True)


def scaled_dot_product_attention(q, k, v, causal=False):
    # every query row scores every key row with a dot product, turns the scores into
    # weights with softmax, and returns the weighted average of the value rows:
    #   attention(q, k, v) = softmax(q @ k^T / sqrt(d)) @ v
    # q is (..., T_q, d), k is (..., T_k, d) and v is (..., T_k, d_v); the output is
    # (..., T_q, d_v), one averaged value row per query. the leading dims (batch, heads)
    # must match.
    # why sqrt(d): a dot product of d random terms grows like sqrt(d), and big scores push
    # softmax towards a single 1 and zeros, where its gradients almost vanish.
    # causal=True lets position i look only at positions 0..i, for models that predict the
    # next token and must not see it: the scores for later positions get -inf added, so
    # softmax gives them weight exactly 0. position i can always see itself, so no row is
    # left with nothing to attend to.
    # built from matmul, transpose, softmax and +, so backward comes from those ops
    if len(q.shape) < 2 or len(k.shape) < 2 or len(v.shape) < 2:
        raise ValueError(f"attention needs q, k, v with at least 2 dims, got {q.shape}, {k.shape}, {v.shape}")
    if not q.shape[:-2] == k.shape[:-2] == v.shape[:-2]:
        raise ValueError(f"q, k, v need the same leading dims, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"q and k need the same last dim, and k and v the same number of rows, "
                         f"got {q.shape}, {k.shape}, {v.shape}")
    d = q.shape[-1]
    scores = (q @ k.transpose()) * (d ** -0.5)
    if causal:
        T = q.shape[-2]
        if k.shape[-2] != T:
            raise ValueError(f"causal attention needs as many queries as keys, got {T} and {k.shape[-2]}")
        # (T, T), broadcast over the leading dims
        mask = tensor([[float("-inf") if j > i else 0.0 for j in range(T)] for i in range(T)])
        scores = scores + mask
    return scores.softmax() @ v


def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    # picks ids from a model's output scores, for generating text one token at a time.
    # logits is (V,) for one pick or (B, V) for one pick per row; returns an int, or a
    # list of B ints. each row becomes probabilities
    #   p_i = softmax(z / temperature)_i
    # temperature 1 samples the model's own distribution, below 1 sharpens it towards the
    # top score, above 1 flattens it towards uniform, and 0 always takes the top score.
    # top_k keeps only the k highest scores (ties go to the lower index) and gives the rest
    # probability 0, so a long tail of unlikely tokens can't be picked by bad luck.
    # top_p (nucleus sampling, Holtzman et al., 2020) keeps the fewest top tokens whose
    # probabilities add up to at least top_p, after temperature and top_k, and scales their
    # probabilities up to add up to 1 again. when the model is sure, that's one or two
    # tokens, and when it isn't, many. top_k keeps the same number either way, so it can cut
    # off real options when the model is unsure and let in unlikely ones when it's sure.
    # the draw: lay the probabilities end to end along [0, 1), pick a uniform random u and
    # return the index whose segment u lands in. segment i has length p_i, so index i is
    # picked with chance p_i.
    # no backward: the result is a choice of index, which doesn't change smoothly with the
    # scores, so there is no gradient to send back. sampling is for generating after
    # training, never part of the loss.
    # rng is anything with a random() method, like random.Random(seed); the default is the
    # random module, so random.seed() makes the picks repeatable
    rng = random if rng is None else rng
    if len(logits.shape) not in (1, 2):
        raise ValueError(f"sample expects logits of shape (V,) or (B, V), got {logits.shape}")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature < float("inf"):
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"top_k must be an int >= 1 or None, got {top_k!r}")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1):
        raise ValueError(f"top_p must be a number in (0, 1] or None, got {top_p!r}")
    V = logits.shape[-1]
    picks = []
    for start in range(0, len(logits.data), V):
        row = logits.data[start:start + V]
        if top_k is not None and top_k < V:
            # sorted is stable, so among equal scores the lower index comes first
            keep = set(sorted(range(V), key=lambda i: -row[i])[:top_k])
            row = [z if i in keep else float("-inf") for i, z in enumerate(row)]
        if max(row) == float("-inf"):
            raise ValueError("sample needs at least one score above -inf in each row")
        if temperature == 0:
            picks.append(row.index(max(row)))
            continue
        probs = tensor([z / temperature for z in row]).softmax().data
        if top_p is not None and top_p < 1:
            # most likely first, and the lower index first on ties
            kept, total = set(), 0.0
            for i in sorted(range(V), key=lambda i: -probs[i]):
                kept.add(i)
                total += probs[i]
                if total >= top_p:
                    break
            probs = [p / total if i in kept else 0.0 for i, p in enumerate(probs)]
        u = rng.random()
        end = 0.0
        pick = None
        for i, p in enumerate(probs):
            end += p
            if u < end:
                pick = i
                break
        if pick is None:
            # rounding can leave the probabilities summing to just under 1, with u past the
            # end; that sliver belongs to the last index that can be picked
            pick = max(i for i, p in enumerate(probs) if p > 0)
        picks.append(pick)
    return picks if len(logits.shape) == 2 else picks[0]


def _cos_any(x):
    # optim's _cos is a Taylor series, which gets less accurate further from 0. cos repeats
    # every 2 pi, so take x mod 2 pi first. on [0, 2 pi] the 20 terms are accurate to about 2e-14
    return _cos(x % (2 * PI))


def sinusoidal_positions(length, dim, base=10000.0):
    # a fixed (length, dim) table with one row per position, to add to the token vectors
    # instead of a learned position embedding (Vaswani et al., 2017). the columns come in
    # pairs, and pair i turns at its own speed w_i:
    #   pe[p, 2i] = sin(p * w_i),  pe[p, 2i + 1] = cos(p * w_i),  w_i = 1 / base^(2i / dim)
    # the first pair turns once every 2 pi ~ 6 positions and the last one about once every
    # 2 pi * base positions, so near positions differ in the fast pairs and far ones in the
    # slow pairs. moving k positions on turns every pair by an angle that depends only on k:
    #   sin((p + k) w) = sin(p w) cos(k w) + cos(p w) sin(k w)
    #   cos((p + k) w) = cos(p w) cos(k w) - sin(p w) sin(k w)
    # so row p + k is row p times a matrix that is the same for every p, and a linear layer
    # can learn to look a fixed number of positions back. the table has no weights, so it
    # works for any length, including ones longer than seen in training.
    # it's a constant: nothing to learn and no backward
    for name, value, low in (("length", length, 1), ("dim", dim, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
    if dim % 2:
        raise ValueError(f"dim must be even, since the columns come in sin/cos pairs, got {dim}")
    if not base > 1:
        raise ValueError(f"base must be > 1, got {base}")
    speeds = [base ** (-2 * i / dim) for i in range(dim // 2)]
    rows = []
    for p in range(length):
        row = []
        for w in speeds:
            # sin(x) = cos(x - pi/2)
            row += [_cos_any(p * w - PI / 2), _cos_any(p * w)]
        rows.append(row)
    return tensor(rows)


class MLP(module):
    def __init__(self, layer_sizes, activation_fns=None, dropout_p=None):
        super().__init__()
        self.layers = []
        self.dropouts = []
        
        num_hidden = len(layer_sizes) - 2
        
        if activation_fns is None:
            self.activation_fns = [lambda x: x.relu()] * num_hidden + [None]
        else:
            self.activation_fns = activation_fns
        
        if dropout_p is None:
            self.dropout_ps = [0.0] * num_hidden
        else:
            self.dropout_ps = dropout_p if isinstance(dropout_p, list) else [dropout_p] * num_hidden

        # one activation per hidden layer, plus an optional last one for the output layer
        if len(self.activation_fns) not in (num_hidden, num_hidden + 1):
            raise ValueError(f"Expected {num_hidden} or {num_hidden + 1} activation_fns, got {len(self.activation_fns)}")
        if len(self.dropout_ps) != num_hidden:
            raise ValueError(f"Expected {num_hidden} dropout rates (one per hidden layer), got {len(self.dropout_ps)}")
        
        for i in range(len(layer_sizes)-1):
            self.layers.append(linear(layer_sizes[i], layer_sizes[i+1]))
            # one entry per hidden layer, None where there is no dropout.
            # kept as plain modules in a list so train() and eval() reach them
            if i < num_hidden:
                p = self.dropout_ps[i]
                self.dropouts.append(dropout(p=p) if p > 0 else None)

    def __call__(self,x):
        out = x

        for i, layer in enumerate(self.layers):
            out = layer(out)

            if i < len(self.layers) - 1:
                act_fn = self.activation_fns[i]
                if act_fn:
                    out = act_fn(out)

                if self.dropouts[i] is not None:
                    out = self.dropouts[i](out)
            elif i < len(self.activation_fns) and self.activation_fns[i]:
                out = self.activation_fns[i](out)
        
        return out

    def __repr__(self):
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "MLP(\n" + "\n".join(lines) + "\n)"
    

class sequential(module):
    # runs layers one after another, each one's output going into the next.
    # a layer can be a module or any function, like lambda x: x.relu()
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(lines) + "\n)"


class conv2d(module):
    # a layer of out_channels square filters, each kernel_size x kernel_size and covering
    # all in_channels, slid over (N, in_channels, H, W) images. one bias per filter, stored
    # as (out_channels, 1, 1) so it broadcasts over every pixel of that filter's output.
    # init is scaled by fan-in like linear, where fan-in is in_channels * kernel_size^2
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        bound = (in_channels * kernel_size * kernel_size) ** -0.5
        count = out_channels * in_channels * kernel_size * kernel_size
        self.w = tensor([random.uniform(-bound, bound) for _ in range(count)], shape=shape)
        self.b = tensor([0.0] * out_channels, shape=(out_channels, 1, 1))

    def __call__(self, x):
        return x.conv2d(self.w, self.stride, self.padding) + self.b

    def __repr__(self):
        O, C, K, _ = self.w.shape
        return f"Conv2d({C}, {O}, kernel_size={K}, stride={self.stride}, padding={self.padding})"


class max_pool2d(module):
    # keeps the largest value in each kernel_size x kernel_size window of every channel
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        return x.max_pool2d(self.kernel_size, self.stride)

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size})"


class avg_pool2d(module):
    # averages each kernel_size x kernel_size window of every channel
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        return x.avg_pool2d(self.kernel_size, self.stride)

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size})"


class flatten(module):
    # joins every dim from start_dim on, so (N, C, H, W) conv outputs become (N, C * H * W)
    # rows for a linear layer. start_dim=1 keeps the batch dim
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def __call__(self, x):
        return x.flatten(self.start_dim)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


class rnn_cell(module):
    # one step of a plain recurrent network. it reads one input x_t (N, input_size) and the
    # hidden state from the step before (N, hidden_size), and returns the next hidden state:
    #   h_t = tanh(x_t @ w_x + h_(t-1) @ w_h + b)
    # the hidden state is the network's memory of everything it has read so far. the same
    # weights are used at every step, so a sequence is read with a plain loop:
    #   h = None
    #   for x_t in sequence:
    #       h = cell(x_t, h)
    # backward through the loop (backprop through time) needs nothing extra: every step adds
    # nodes to the graph, and the shared weights collect grads from all the steps
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # uniform(-1/sqrt(hidden_size), 1/sqrt(hidden_size)) weights, like pytorch
        bound = hidden_size ** -0.5
        self.w_x = tensor([[random.uniform(-bound, bound) for _ in range(hidden_size)] for _ in range(input_size)])
        self.w_h = tensor([[random.uniform(-bound, bound) for _ in range(hidden_size)] for _ in range(hidden_size)])
        self.b = tensor([[0.0] * hidden_size])

    def __call__(self, x, h=None):
        input_size, hidden_size = self.w_x.shape
        if len(x.shape) != 2 or x.shape[1] != input_size:
            raise ValueError(f"rnn_cell expects input shape (N, {input_size}), got {x.shape}")
        if h is None:
            # nothing read yet: start from zeros
            h = tensor([0.0] * (x.shape[0] * hidden_size), shape=(x.shape[0], hidden_size))
        if h.shape != (x.shape[0], hidden_size):
            raise ValueError(f"rnn_cell expects hidden shape ({x.shape[0]}, {hidden_size}), got {h.shape}")
        return (x @ self.w_x + h @ self.w_h + self.b).tanh()

    def __repr__(self):
        return f"RNNCell({self.w_x.shape[0]}, {self.w_x.shape[1]})"


class lstm_cell(module):
    # one step of a long short-term memory network (Hochreiter and Schmidhuber, 1997, with
    # the forget gate from Gers et al., 2000). besides the hidden state h it carries a cell
    # state c, and four gates, computed from x_t and h_(t-1), decide how c changes:
    #   i = sigmoid(...)   input gate, how much of the candidate to write into c
    #   f = sigmoid(...)   forget gate, how much of the old c to keep
    #   g = tanh(...)      candidate values to write
    #   o = sigmoid(...)   output gate, how much of c to show in h
    #   c_t = f * c_(t-1) + i * g
    #   h_t = o * tanh(c_t)
    # each "..." is x_t @ w_x + h_(t-1) @ w_h + b. all four are computed by one matmul into
    # (N, 4 * hidden_size) columns, in the order i, f, g, o as in pytorch, then cut apart.
    # from one step to the next c is only multiplied by f and added to, with no tanh in
    # between, so dc_t/dc_(t-1) = f. while f stays near 1 the gradient goes back through
    # many steps without shrinking. in rnn_cell it gets multiplied by tanh' and w_h at
    # every step instead, which usually makes it fade over long sequences.
    # use it like rnn_cell, passing the state (h, c) along:
    #   state = None
    #   for x_t in sequence:
    #       h, c = cell(x_t, state)
    #       state = (h, c)
    def __init__(self, input_size, hidden_size):
        super().__init__()
        bound = hidden_size ** -0.5
        self.w_x = tensor([[random.uniform(-bound, bound) for _ in range(4 * hidden_size)] for _ in range(input_size)])
        self.w_h = tensor([[random.uniform(-bound, bound) for _ in range(4 * hidden_size)] for _ in range(hidden_size)])
        # the forget gate's bias starts at 1, so f starts near sigmoid(1) = 0.73 and the cell
        # keeps most of what it holds until training says otherwise (Jozefowicz et al., 2015)
        H = hidden_size
        self.b = tensor([[0.0] * H + [1.0] * H + [0.0] * (2 * H)])

    def __call__(self, x, state=None):
        input_size, H = self.w_x.shape[0], self.w_h.shape[0]
        if len(x.shape) != 2 or x.shape[1] != input_size:
            raise ValueError(f"lstm_cell expects input shape (N, {input_size}), got {x.shape}")
        N = x.shape[0]
        if state is None:
            # nothing read yet: start from zeros
            zeros = tensor([0.0] * (N * H), shape=(N, H))
            state = (zeros, zeros)
        h, c = state
        for name, t in (("hidden", h), ("cell", c)):
            if t.shape != (N, H):
                raise ValueError(f"lstm_cell expects {name} shape ({N}, {H}), got {t.shape}")

        gates = x @ self.w_x + h @ self.w_h + self.b
        i = gates[:, :H].sigmoid()
        f = gates[:, H:2 * H].sigmoid()
        g = gates[:, 2 * H:3 * H].tanh()
        o = gates[:, 3 * H:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        return h, c

    def __repr__(self):
        return f"LSTMCell({self.w_x.shape[0]}, {self.w_h.shape[0]})"


class gru_cell(module):
    # one step of a gated recurrent unit (Cho et al., 2014). it has gates like lstm_cell
    # but no separate cell state, only h, and three sets of columns instead of four:
    #   r = sigmoid(x_t @ w_x[r] + b_x[r] + h_(t-1) @ w_h[r] + b_h[r])      reset gate
    #   z = sigmoid(x_t @ w_x[z] + b_x[z] + h_(t-1) @ w_h[z] + b_h[z])      update gate
    #   n = tanh(x_t @ w_x[n] + b_x[n] + r * (h_(t-1) @ w_h[n] + b_h[n]))   candidate
    #   h_t = (1 - z) * n + z * h_(t-1)
    # z decides, feature by feature, whether to keep the old h or take the candidate n.
    # where z is near 1, h passes through nearly unchanged and so does its gradient, like
    # the lstm's cell state with f near 1. r decides how much of the old h goes into n.
    # this is pytorch's version, where r multiplies h_(t-1) @ w_h[n] after the matmul. the
    # paper applies it before, (r * h_(t-1)) @ w_h[n]. with r on the outside, the h side
    # needs its own bias inside the brackets, so there are two biases. for r and z only
    # b_x + b_h matters. used like rnn_cell: h = cell(x_t, h), starting from h = None
    def __init__(self, input_size, hidden_size):
        super().__init__()
        bound = hidden_size ** -0.5
        H = hidden_size
        self.w_x = tensor([[random.uniform(-bound, bound) for _ in range(3 * H)] for _ in range(input_size)])
        self.w_h = tensor([[random.uniform(-bound, bound) for _ in range(3 * H)] for _ in range(H)])
        self.b_x = tensor([[0.0] * (3 * H)])
        self.b_h = tensor([[0.0] * (3 * H)])

    def __call__(self, x, h=None):
        input_size, H = self.w_x.shape[0], self.w_h.shape[0]
        if len(x.shape) != 2 or x.shape[1] != input_size:
            raise ValueError(f"gru_cell expects input shape (N, {input_size}), got {x.shape}")
        N = x.shape[0]
        if h is None:
            # nothing read yet: start from zeros
            h = tensor([0.0] * (N * H), shape=(N, H))
        if h.shape != (N, H):
            raise ValueError(f"gru_cell expects hidden shape ({N}, {H}), got {h.shape}")

        gx = x @ self.w_x + self.b_x
        gh = h @ self.w_h + self.b_h
        r = (gx[:, :H] + gh[:, :H]).sigmoid()
        z = (gx[:, H:2 * H] + gh[:, H:2 * H]).sigmoid()
        n = (gx[:, 2 * H:] + r * gh[:, 2 * H:]).tanh()
        return (1.0 - z) * n + z * h

    def __repr__(self):
        return f"GRUCell({self.w_x.shape[0]}, {self.w_h.shape[0]})"


class multi_head_attention(module):
    # runs num_heads attentions side by side, each on its own slice of dim // num_heads
    # features, so different heads can look for different things (one might follow the
    # previous token, another a matching word further back). for x of shape (B, T, dim):
    #   1. three linear layers turn every position into a query, a key and a value
    #   2. each is cut into heads: (B, T, dim) -> (B, T, heads, head_dim) -> (B, heads, T, head_dim)
    #   3. scaled dot-product attention runs in every head at once, heads acting like a batch dim
    #   4. the heads go back side by side, (B, T, dim), and a last linear layer mixes them
    # it is self-attention: the queries, keys and values all come from the same x
    def __init__(self, dim, num_heads):
        super().__init__()
        if num_heads < 1 or dim % num_heads != 0:
            raise ValueError(f"dim {dim} must split evenly into num_heads {num_heads}")
        self.num_heads = num_heads
        self.q_proj = linear(dim, dim)
        self.k_proj = linear(dim, dim)
        self.v_proj = linear(dim, dim)
        self.out_proj = linear(dim, dim)

    def __call__(self, x, causal=False):
        dim = self.q_proj.w.shape[0]
        if len(x.shape) != 3 or x.shape[2] != dim:
            raise ValueError(f"multi_head_attention expects shape (B, T, {dim}), got {x.shape}")
        B, T, _ = x.shape
        H = self.num_heads

        def split_heads(t):
            return t.reshape(B, T, H, dim // H).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        out = scaled_dot_product_attention(q, k, v, causal)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, dim))

    def __repr__(self):
        return f"MultiHeadAttention(dim={self.q_proj.w.shape[0]}, num_heads={self.num_heads})"


class transformer_block(module):
    # one layer of a GPT-style transformer, for x of shape (B, T, dim):
    #   x = x + attn(ln1(x))    positions pass information to each other
    #   x = x + mlp(ln2(x))     each position works on what it gathered, on its own
    # the mlp widens every position to hidden features (4 * dim by default), applies gelu
    # and narrows back to dim.
    # the "x +" paths are residual connections. each sub-layer only adds a change to x,
    # and since d(x + f(x))/dx = 1 + df/dx, the gradient always has a direct route back to
    # the input, so many blocks can be stacked and still train. layer norm comes before
    # each sub-layer (pre-norm, as in GPT-2), so the residual path itself is never
    # normalized. the output has the same shape as x, ready for the next block.
    # built from existing layers, so backward comes from their ops
    def __init__(self, dim, num_heads, hidden=None):
        super().__init__()
        hidden = 4 * dim if hidden is None else hidden
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError(f"hidden must be an int >= 1, got {hidden!r}")
        self.ln1 = layer_norm(dim)
        self.attn = multi_head_attention(dim, num_heads)
        self.ln2 = layer_norm(dim)
        self.mlp = MLP([dim, hidden, dim], activation_fns=[lambda x: x.gelu()])

    def __call__(self, x, causal=False):
        x = x + self.attn(self.ln1(x), causal)
        return x + self.mlp(self.ln2(x))

    def __repr__(self):
        return (f"TransformerBlock(dim={self.ln1.gamma.shape[0]}, num_heads={self.attn.num_heads}, "
                f"hidden={self.mlp.layers[0].w.shape[1]})")


class embedding(module):
    # a lookup table with one learned vector (row) per id, for tokens or categories.
    # given integer ids of any shape, returns their rows, shape ids.shape + (dim,).
    # it gives the same result as one_hot(ids) @ weight without building the one-hot
    # vectors: forward copies rows, and backward adds each output row's grad into the row
    # it came from (more than once if an id repeats). indexing with a list already does both
    def __init__(self, num_embeddings, dim):
        super().__init__()
        # standard normal init, like pytorch
        self.weight = tensor([[random.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(num_embeddings)])

    def __call__(self, ids):
        ids = ids if isinstance(ids, tensor) else tensor(ids)
        n, dim = self.weight.shape
        for i in ids.data:
            if isinstance(i, bool) or not float(i).is_integer() or not 0 <= i < n:
                raise ValueError(f"Embedding id {i!r} is not an int in 0..{n - 1}")
        rows = self.weight[[int(i) for i in ids.data]]
        return rows.reshape(ids.shape + (dim,))

    def __repr__(self):
        return f"Embedding({self.weight.shape[0]}, {self.weight.shape[1]})"


class layer_norm(module):
    # normalizes each example's features (the last dim) to mean 0 and variance 1, then
    # scales by gamma and shifts by beta, both learned, one per feature. they start at 1
    # and 0, so the layer begins as a plain normalization and can learn to undo it
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = tensor([1.0] * dim)
        self.beta = tensor([0.0] * dim)

    def __call__(self, x):
        if not x.shape or x.shape[-1] != self.gamma.shape[0]:
            raise ValueError(f"layer_norm expects a last dim of {self.gamma.shape[0]}, got shape {x.shape}")
        return x.layer_norm(self.eps) * self.gamma + self.beta

    def __repr__(self):
        return f"LayerNorm({self.gamma.shape[0]}, eps={self.eps})"


class rms_norm(module):
    # divides each example's features (the last dim) by their root mean square, then
    # scales by a learned gamma, one per feature, starting at 1. there's no beta: the
    # norm doesn't move the mean to 0, so there's nothing to shift back
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = tensor([1.0] * dim)

    def __call__(self, x):
        if not x.shape or x.shape[-1] != self.gamma.shape[0]:
            raise ValueError(f"rms_norm expects a last dim of {self.gamma.shape[0]}, got shape {x.shape}")
        return x.rms_norm(self.eps) * self.gamma

    def __repr__(self):
        return f"RMSNorm({self.gamma.shape[0]}, eps={self.eps})"


class batch_norm(module):
    # normalizes each feature over the batch: every column of an (N, C) input gets the
    # batch mean of that column subtracted and is divided by the batch std, then scaled by
    # gamma and shifted by beta (learned, one per feature). unlike layer_norm, an example's
    # output depends on the other examples in its batch.
    # a test batch may hold a single example, so training also keeps running averages,
    #   running = (1 - momentum) * running + momentum * batch value,
    # and eval() mode normalizes with those instead. the running variance is updated with
    # the unbiased batch variance (divided by N - 1 instead of N), like pytorch does
    buffer_names = ("running_mean", "running_var")

    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        if not 0 <= momentum <= 1:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.momentum = momentum
        self.eps = eps
        self.gamma = tensor([1.0] * dim)
        self.beta = tensor([0.0] * dim)
        self.running_mean = tensor([0.0] * dim)
        self.running_var = tensor([1.0] * dim)

    def __call__(self, x):
        C = self.gamma.shape[0]
        if len(x.shape) != 2 or x.shape[1] != C:
            raise ValueError(f"batch_norm expects shape (N, {C}), got {x.shape}")
        if self.training:
            N = x.shape[0]
            if N < 2:
                raise ValueError("batch_norm needs at least 2 rows in training mode")
            # built from mean and var, so backward comes from their gradients,
            # including the path through the batch statistics
            mean = x.mean(axis=0, keepdim=True)
            var = x.var(axis=0, keepdim=True)
            m = self.momentum
            for j in range(C):
                self.running_mean.data[j] = (1 - m) * self.running_mean.data[j] + m * mean.data[j]
                unbiased = var.data[j] * N / (N - 1)
                self.running_var.data[j] = (1 - m) * self.running_var.data[j] + m * unbiased
        else:
            # copies, so the running stats stay out of the graph
            mean = tensor(self.running_mean.data, shape=(1, C))
            var = tensor(self.running_var.data, shape=(1, C))
        x_hat = (x - mean) * (var + self.eps) ** -0.5
        return x_hat * self.gamma + self.beta

    def __repr__(self):
        return f"BatchNorm({self.gamma.shape[0]}, momentum={self.momentum}, eps={self.eps})"


class dropout(module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"dropout p must be in [0, 1), got {p}")
        self.p = p

    def __call__(self, x):
        if not self.training:
            return x
        scale = 1.0 / (1.0 - self.p)
        mask_data = [scale if random.random() > self.p else 0.0 for _ in range(len(x.data))]

        # label the multiply, not the mask, so the graph shows x and mask going into dropout
        mask = tensor(mask_data, shape=x.shape)
        out = x * mask
        out._op = 'dropout'
        return out
    
    def __repr__(self):
        return f"Dropout(p={self.p})"