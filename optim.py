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

class Adagrad(optimizer):
    # gives every weight its own step size by dividing by the size of all its gradients so far:
    #   s = s + g^2
    #   w -= lr * g / (sqrt(s) + eps)
    # weights with big gradients get small steps and weights with small ones get big steps.
    # s only grows, so the steps keep shrinking: with a steady gradient the t-th step is
    # lr / sqrt(t). a weight that rarely gets a gradient, like the embedding row of a rare
    # word, keeps a small s and so keeps taking big steps when it does get one (Duchi et al.,
    # 2011). on long runs the shrinking can stop training too early, which RMSprop fixes.
    # eps turns 0 / 0 into a step of 0 while s is still 0, and otherwise only changes
    # steps for gradients about as small as eps itself
    def __init__(self, parameters, lr=0.01, eps=1e-10):
        super().__init__(parameters, lr)
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = eps
        self.sum_sq = [[0.0] * len(p.data) for p in self.parameters]

    def step(self):
        for p, s in zip(self.parameters, self.sum_sq):
            for i in range(len(p.data)):
                g = p.grad[i]
                s[i] += g * g
                p.data[i] -= self.lr * g / (s[i] ** 0.5 + self.eps)


class RMSprop(optimizer):
    # like Adagrad, but it divides by a running average of recent squared gradients
    # instead of the sum of all of them (Tieleman and Hinton, 2012):
    #   v = alpha * v + (1 - alpha) * g^2
    #   w -= lr * g / (sqrt(v) + eps)
    # sqrt(v) is about the size of recent gradients, so each step is about lr whatever the
    # gradient's scale. the average forgets old gradients (alpha = 0.99 mostly reflects the
    # last 100 steps or so), so the steps don't shrink forever and follow the gradients
    # when they change. v starts at 0 and isn't corrected for that, so the first steps are
    # bigger than lr: the first one is lr / sqrt(1 - alpha), 10 * lr with alpha = 0.99.
    # Adam is RMSprop plus momentum and that correction
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        if not 0 <= alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.alpha = alpha
        self.eps = eps
        self.v = [[0.0] * len(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.v):
            for i in range(len(p.data)):
                g = p.grad[i]
                v[i] = self.alpha * v[i] + (1 - self.alpha) * g * g
                p.data[i] -= self.lr * g / (v[i] ** 0.5 + self.eps)

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

PI = 3.141592653589793

def _cos(x, terms=20):
    # Taylor series: cos(x) = 1 - x^2/2! + x^4/4! - ...
    # each term is the one before times -x^2 / ((2n+1)(2n+2)).
    # only used with x in [0, pi], where 20 terms are accurate to about 1e-15
    result = 0.0
    term = 1.0
    for n in range(terms):
        result += term
        term *= -x * x / ((2 * n + 1) * (2 * n + 2))
    return result


class lr_scheduler:
    # changes optimizer.lr as training goes on. call step() once per epoch, after
    # optimizer.step(). each subclass gives the lr for an epoch number with lr_at(epoch).
    # epoch 0 uses the lr the optimizer was made with
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.lr_at(self.epoch)

class StepLR(lr_scheduler):
    # multiplies the lr by gamma every step_size epochs: lr = base_lr * gamma ** (epoch // step_size)
    # with lr=0.1, step_size=10, gamma=0.5 that is 0.1 for epochs 0-9, 0.05 for 10-19, ...
    def __init__(self, optimizer, step_size, gamma=0.1):
        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError(f"step_size must be an int >= 1, got {step_size}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def lr_at(self, epoch):
        return self.base_lr * self.gamma ** (epoch // self.step_size)

class CosineAnnealingLR(lr_scheduler):
    # lowers the lr from base_lr to eta_min along half a cosine wave over T_max epochs:
    # lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    # cos goes from 1 to -1, so the lr starts at base_lr and ends at eta_min. it falls slowly
    # at first, fastest in the middle, and slowly again at the end.
    # after T_max epochs the lr stays at eta_min
    def __init__(self, optimizer, T_max, eta_min=0.0):
        if T_max <= 0:
            raise ValueError(f"T_max must be > 0, got {T_max}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def lr_at(self, epoch):
        epoch = min(epoch, self.T_max)
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + _cos(PI * epoch / self.T_max)) / 2
