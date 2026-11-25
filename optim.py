from contextlib import contextmanager

try:
    from .engine import PI, tensor
except ImportError:
    # run as a script from this folder, not imported as a package
    from engine import PI, tensor


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


class Lion(optimizer):
    # keeps a momentum like Adam but steps by its sign, so every weight moves by exactly lr
    # each step (Chen et al., 2023, found by a program search over optimizers):
    #   update = sign(beta1 * m + (1 - beta1) * g)
    #   w = w * (1 - lr * weight_decay) - lr * update
    #   m = beta2 * m + (1 - beta2) * g
    # the step mixes in more of the new gradient (beta1 = 0.9) than the momentum it keeps
    # for later (beta2 = 0.99), so it reacts to the latest gradient while m stays smooth.
    # it stores one list per parameter instead of Adam's two. since every step has size lr,
    # it needs a smaller lr than Adam, the paper suggests 3 to 10 times smaller, and a
    # weight_decay that much bigger to keep lr * weight_decay the same. the decay is
    # decoupled like AdamW's. a weight whose update is exactly 0 isn't moved, apart from decay
    def __init__(self, parameters, lr=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.0):
        super().__init__(parameters, lr)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"beta1 and beta2 must be in [0, 1), got {beta1} and {beta2}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m = [[0.0] * len(p.data) for p in self.parameters]

    def step(self):
        for p, m in zip(self.parameters, self.m):
            for i in range(len(p.data)):
                g = p.grad[i]
                u = self.beta1 * m[i] + (1 - self.beta1) * g
                sign = (u > 0) - (u < 0)
                p.data[i] = p.data[i] * (1 - self.lr * self.weight_decay) - self.lr * sign
                m[i] = self.beta2 * m[i] + (1 - self.beta2) * g


class EMA:
    # an exponential moving average of a model's weights. after every optimizer step,
    #   average = decay * average + (1 - decay) * w
    # the weights at the end of training jump around with the noise of the last few batches;
    # the average smooths that out over about 1 / (1 - decay) steps (1000 for 0.999), and
    # often scores better on held-out data. training still updates the real weights; the
    # average is only used to evaluate or save:
    #   ema = EMA(model, decay=0.999)
    #   for each batch: loss.backward(); opt.step(); ema.update()
    #   with ema.averaged():
    #       evaluate(model)    # the model holds the averaged weights in here
    # the average starts as a copy of the weights when EMA is made, so early on it is pulled
    # towards the starting weights. buffers like batch_norm's running stats aren't averaged:
    # inside averaged() the model uses its current ones
    def __init__(self, model, decay=0.999):
        if not 0 <= decay < 1:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.model = model
        self.decay = decay
        self.average = {name: p.data[:] for name, p in model.named_parameters()}
        self._swapped = False

    def update(self):
        d = self.decay
        for name, p in self.model.named_parameters():
            avg = self.average[name]
            for i in range(len(avg)):
                avg[i] = d * avg[i] + (1 - d) * p.data[i]

    @contextmanager
    def averaged(self):
        # puts the averaged values into the model's own tensors and puts the trained values
        # back afterwards, even if the body raises. the values are copied in place, so an
        # optimizer holding those tensors keeps working
        if self._swapped:
            raise RuntimeError("already inside averaged()")
        params = self.model.named_parameters()
        trained = {name: p.data[:] for name, p in params}
        for name, p in params:
            p.data[:] = self.average[name]
        self._swapped = True
        try:
            yield self.model
        finally:
            for name, p in params:
                p.data[:] = trained[name]
            self._swapped = False


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
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + tensor._taylor_cos(PI * epoch / self.T_max)) / 2

class LinearWarmup(lr_scheduler):
    # starts with a small lr and raises it in equal steps to the optimizer's lr:
    #   lr = base_lr * (epoch + 1) / (warmup_steps + 1)     while epoch < warmup_steps
    # that's base_lr / (warmup_steps + 1) at epoch 0, never 0, so no step is wasted, and
    # base_lr at epoch warmup_steps. from there on it's base_lr, or the lr of the `after`
    # schedule counted from that schedule's own epoch 0:
    #   LinearWarmup(opt, 100, after=CosineAnnealingLR(opt, T_max=900))
    # make `after` first, so it reads the full lr, and only call the warmup's step().
    # early on the weights are random, and Adam's and RMSprop's averages of the squared
    # gradients come from only a few steps (RMSprop's first step is 10 * lr), so full-size
    # steps can throw the weights far off. small first steps give those averages time to
    # settle. transformers have been trained with warmup since Vaswani et al. (2017).
    # unlike the other schedules this one sets the lr as soon as it's made, so epoch 0
    # already uses the small lr
    def __init__(self, optimizer, warmup_steps, after=None):
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
            raise ValueError(f"warmup_steps must be an int >= 1, got {warmup_steps!r}")
        if after is not None and (not isinstance(after, lr_scheduler) or after.optimizer is not optimizer):
            raise ValueError("after must be a schedule for the same optimizer")
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.after = after
        optimizer.lr = self.lr_at(0)

    def lr_at(self, epoch):
        if epoch < self.warmup_steps:
            return self.base_lr * (epoch + 1) / (self.warmup_steps + 1)
        if self.after is None:
            return self.base_lr
        return self.after.lr_at(epoch - self.warmup_steps)


class ReduceLROnPlateau:
    # lowers the lr when a number you're watching, usually the val loss, stops going down.
    # call step(val_loss) once per epoch. a value counts as better if it beats the best so
    # far by more than threshold times the best's size:
    #   better = value < best - threshold * |best|
    # after more than `patience` epochs in a row that aren't better, the lr is multiplied
    # by factor (but not below min_lr) and the count starts over. with patience = 2 the
    # first 2 epochs without a better value are allowed, and the lr drops after the 3rd.
    # useful when you don't know beforehand how long training will take: the lr stays high
    # while it helps and drops once the loss stops falling, instead of on a fixed schedule.
    # pytorch compares with best * (1 - threshold), which for a negative best lets a worse
    # value count as better; |best| avoids that.
    # it isn't an lr_scheduler subclass: it has no lr for a given epoch, it reacts to values
    def __init__(self, optimizer, factor=0.1, patience=10, threshold=1e-4, min_lr=0.0):
        if not 0 < factor < 1:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError(f"patience must be an int >= 0, got {patience!r}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        # None until the first value comes in. starting at inf wouldn't work: inf - threshold * inf
        # is inf - inf = nan, and nothing is smaller than nan
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None or value < self.best - self.threshold * abs(self.best):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
            self.bad_epochs = 0
