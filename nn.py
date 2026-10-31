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

    def parameters(self):
        # every tensor stored as an attribute (or in a list), plus the parameters of every
        # sub-module, so a new layer only has to store its tensors. a tensor shared by two
        # layers is listed once, so the optimizer doesn't update it twice per step
        params, seen = [], set()
        for v in vars(self).values():
            for item in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(item, tensor):
                    found = [item]
                elif isinstance(item, module):
                    found = item.parameters()
                else:
                    continue
                for p in found:
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
        return params
    
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