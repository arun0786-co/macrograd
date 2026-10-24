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

    def parameters(self):
        return []
    
    def train(self):
        self.training = True
        for v in vars(self).values():
            if isinstance(v, module):
                v.train()
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, module):
                        item.train()
    
    def eval(self):
        self.training = False
        for v in vars(self).values():
            if isinstance(v, module):
                v.eval()
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, module):
                        item.eval()
    
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
    
    def parameters(self):
        return [self.w, self.b]
    
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

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "MLP(\n" + "\n".join(lines) + "\n)"
    

class dropout(module):
    def __init__(self, p=0.5):
        super().__init__()
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
    
    def parameters(self):
        return []
    
    def __repr__(self):
        return f"Dropout(p={self.p})"