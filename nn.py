import json
import random
try:
    from .engine import tensor
except ImportError:
    # run as a script from this folder, not imported as a package
    from engine import tensor

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
        out = x.matmul(self.w) + self.b
        return out
    
    def __repr__(self):
        return f"Linear(in_features={self.w.shape[0]}, out_features={self.w.shape[1]})"


# weight init. each output of x @ w adds up fan_in terms w * x, so its variance is
# fan_in * var(w) * E[x^2]. picking var(w) to cancel fan_in keeps activations the same
# size from layer to layer instead of growing or shrinking with depth.
# these fill a weight in place and return it, e.g. kaiming_normal(layer.w). weights here
# are (fan_in, fan_out), the way linear stores them (inputs, outputs)

def _fans(w):
    if len(w.shape) != 2:
        raise ValueError(f"init expects a 2-d (fan_in, fan_out) weight, got shape {w.shape}")
    return w.shape[0], w.shape[1]


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