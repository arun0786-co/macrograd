import random 
from engine import tensor

class module:
    def zero_grad(self):
        for p in self.parameters():
            for i in range(len(p.data)):
                p.grad[i] = 0.0

    def parameters(self):
        return []
    
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
        w_data = [[random.uniform(-1,1) for _ in range(nout)] for _ in range(nin)]
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
    def __init__(self, layer_sizes):
        
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(linear(layer_sizes[i], layer_sizes[i+1   ]))

    def __call__(self,x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = out.relu()
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        lines = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "MLP(\n" + "\n".join(lines) + "\n)"
    