class optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            for i in range(len(p.data)):
                p.grad[i] = 0.0

class SGD(optimizer):
    # weight_decay adds weight_decay * w to each gradient, which pulls weights toward 0.
    # momentum keeps a running velocity, v = momentum * v + g, and steps along v instead
    # of g, so steps that keep pointing the same way build up speed and noisy ones cancel.
    # momentum=0 and weight_decay=0 is plain SGD: w -= lr * g
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [[0.0] * len(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            for i in range(len(p.data)):
                g = p.grad[i] + self.weight_decay * p.data[i]
                v[i] = self.momentum * v[i] + g
                p.data[i] -= self.lr * v[i]

class Adam(optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0 
        
       
        self.m = [[0.0] * len(p.data) for p in self.parameters]
        self.v = [[0.0] * len(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        
        
        for p_idx, p in enumerate(self.parameters):
            
            for i in range(len(p.data)):
                g = p.grad[i]
                
                
                self.m[p_idx][i] = self.b1 * self.m[p_idx][i] + (1 - self.b1) * g
                
                self.v[p_idx][i] = self.b2 * self.v[p_idx][i] + (1 - self.b2) * (g ** 2)

              
                m_hat = self.m[p_idx][i] / (1 - self.b1 ** self.t)
                v_hat = self.v[p_idx][i] / (1 - self.b2 ** self.t)

                p.data[i] -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)


class AdamW(Adam):
    # Adam with decoupled weight decay: each step first shrinks every weight by
    # (1 - lr * weight_decay), then takes the normal Adam step. adding weight_decay * w
    # to the gradient instead (L2) would get divided by Adam's sqrt(v) like the rest of
    # the gradient, so weights with large gradients would barely be decayed.
    # the Adam step only reads the gradient, so shrinking first doesn't change it
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, beta1, beta2, eps)
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        self.weight_decay = weight_decay

    def step(self):
        for p in self.parameters:
            for i in range(len(p.data)):
                p.data[i] *= 1 - self.lr * self.weight_decay
        super().step()


def clip_grad_norm(parameters, max_norm):
    # treats all the gradients as one long vector. if its length (L2 norm) is over max_norm,
    # every gradient is scaled down by the same factor so the length becomes max_norm.
    # the direction stays the same, only the step gets shorter, which stops one huge
    # gradient from throwing the weights far away. call it after backward(), before step().
    # returns the norm from before clipping, which is useful to print
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    parameters = list(parameters)
    total = sum(g * g for p in parameters for g in p.grad) ** 0.5
    if total > max_norm:
        scale = max_norm / total
        for p in parameters:
            for i in range(len(p.grad)):
                p.grad[i] *= scale
    return total