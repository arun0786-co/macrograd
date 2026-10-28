class tensor:
    def __init__(self, data, shape=None, children=(), op=''):
        if shape is None:
            flat_data, inferred_shape = self._parse_input(data)
            self.data = flat_data
            self.shape = inferred_shape
        else:
            self.data = list(data)
            self.shape = tuple(shape)

        size = 1
        for dim in self.shape:
            size *= dim
        if size != len(self.data):
            raise ValueError(f"{len(self.data)} values don't fit shape {self.shape}")

        self.grad = [0.0] * len(self.data)
        
        self.strides = self._compute_strides(self.shape)
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None

    @staticmethod
    def _taylor_exp(x, terms=15):
        # the series is only accurate near 0, so use exp(-x) = 1/exp(x)
        # and exp(x) = exp(x/2)^2 to bring x into [0, 0.5] first
        if x < 0:
            return 1.0 / tensor._taylor_exp(-x, terms)
        if x > 710.0:
            return float('inf')

        halvings = 0
        while x > 0.5:
            x /= 2.0
            halvings += 1

        result = 1.0
        current_term = 1.0

        for i in range(1, terms):
            current_term *= (x / i)
            result += current_term

        for _ in range(halvings):
            result *= result

        return result

    def exp(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self._taylor_exp(self.data[i])
            
        out = tensor(out_data, shape=self.shape, children=(self,), op='exp')

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.data[i] * out.grad[i]
                
        out._backward = _backward
        return out


    @staticmethod
    def _parse_input(data):
        if not isinstance(data, (list, tuple)):
            return [data], ()
            
        if not data:
            return [], (0,)
            
        if isinstance(data[0], (list, tuple)):
            flat_list = []
            inner_shape = None
            
            for sub_list in data:
                sub_flat, sub_shape = tensor._parse_input(sub_list)
                
                if inner_shape is None:
                    inner_shape = sub_shape
                elif inner_shape != sub_shape:
                    raise ValueError(f"Irregular tensor shape! Expected {inner_shape}, got {sub_shape}")
                    
                flat_list.extend(sub_flat)
                
            return flat_list, (len(data),) + inner_shape
            
        else:
            if any(isinstance(v, (list, tuple)) for v in data):
                raise ValueError("Irregular tensor shape! Got a list next to a number")
            return list(data), (len(data),)



    def __repr__(self):
        return f"tensor(data={self.data}, shape={self.shape})"



    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor([other] * len(self.data), shape=self.shape)
        if isinstance(other, tensor) and self.shape != other.shape:
            shape = self._broadcast_shape(self.shape, other.shape)
            return self.expand(shape) + other.expand(shape)

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} + {other.shape}")

        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self.data[i] + other.data[i]
    
        out = tensor(out_data, shape=self.shape, children=(self, other), op="+")
        
        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += 1.0 * out.grad[i]
                other.grad[i] += 1.0 * out.grad[i]
                
        out._backward = _backward
        return out


    def __mul__(self, other):
        if isinstance(other, (int,float)):
            other = tensor([other] * len(self.data),shape=self.shape)
        if isinstance(other, tensor) and self.shape != other.shape:
            shape = self._broadcast_shape(self.shape, other.shape)
            return self.expand(shape) * other.expand(shape)

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} * {other.shape}")
        
        out_data = [0.0] * len(self.data) 
        for i in range(len(self.data)):
            out_data[i] = self.data[i] * other.data[i]
        out = tensor(out_data, shape=self.shape, children=(self, other), op="*")

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]
        
        out._backward = _backward
        return out
    
    def matmul(self, other):
        if len(self.shape) < 2 or len(other.shape) < 2:
            raise ValueError("matmul requires atleast 2D tensors")
        
        M, N = self.shape[-2], self.shape[-1]
        N2, P = other.shape[-2], other.shape[-1]

        if N != N2:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}")
        
        batch_shape = self.shape[:-2]
        if batch_shape != other.shape[:-2]:
            raise ValueError(f"Batch shapes must match. Got {batch_shape} and {other.shape[:-2]}")

        
        batches = 1
        for dim in batch_shape:
            batches *= dim
            
        
        out_shape = batch_shape + (M, P)
        out_data = [0.0] * (batches * M * P)

        
        stride_A_row, stride_A_col = self.strides[-2], self.strides[-1]
        stride_B_row, stride_B_col = other.strides[-2], other.strides[-1]

        
        matrix_size_A = M * N
        matrix_size_B = N * P
        matrix_size_out = M * P

       
        for b in range(batches):
            
            offset_A = b * matrix_size_A
            offset_B = b * matrix_size_B
            offset_out = b * matrix_size_out

            
            for i in range(M):          
                for j in range(P):      
                    dot_sum = 0.0
                    for k in range(N):  
                        
                        a_val = self.data[offset_A + (i * stride_A_row) + (k * stride_A_col)]
                        b_val = other.data[offset_B + (k * stride_B_row) + (j * stride_B_col)]
                        
                        dot_sum += a_val * b_val
                        
                    out_data[offset_out + (i * P) + j] = dot_sum
                    
        out = tensor(out_data, shape=out_shape, children=(self, other), op="@")
        
        
        def _backward():
            for b in range(batches):
                offset_A = b * matrix_size_A
                offset_B = b * matrix_size_B
                offset_out = b * matrix_size_out

                for i in range(M):          
                    for j in range(P):      
                        
                        grad_out = out.grad[offset_out + (i * P) + j]
                        
                        for k in range(N):  
                            
                            idx_A = offset_A + (i * stride_A_row) + (k * stride_A_col)
                            idx_B = offset_B + (k * stride_B_row) + (j * stride_B_col)
                            
                            
                            self.grad[idx_A] += grad_out * other.data[idx_B]
                            
                            other.grad[idx_B] += self.data[idx_A] * grad_out
            
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def backward(self):
        # iterative post-order walk, so deep graphs don't hit python's recursion limit
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))
        self.grad = [1.0] * len(self.data)
        for node in reversed(topo):
            node._backward()
    
    def _compute_strides(self, shape):
        strides = [0] * len(shape)
        current_stride = 1
        
        for i in reversed(range(len(shape))):
            strides[i] = current_stride
            current_stride *= shape[i]
            
        return tuple(strides)
        
    def get_element(self, *indices):
        if len(indices) != len(self.shape) or any(not 0 <= i < d for i, d in zip(indices, self.shape)):
            raise IndexError(f"Index {indices} out of range for shape {self.shape}")
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        return self.data[flat_index]

    def reshape(self, *shape):
        # data is stored flat, so reshaping only changes the shape and the values keep
        # their order. backward is the same: each grad goes straight back to its position.
        # accepts x.reshape(2, 3) or x.reshape((2, 3)); one dim can be -1 and is inferred
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if shape.count(-1) > 1 or any(d < -1 for d in shape):
            raise ValueError(f"Invalid shape {shape}")

        known = 1
        for d in shape:
            if d != -1:
                known *= d
        if -1 in shape:
            if known == 0 or len(self.data) % known != 0:
                raise ValueError(f"Cannot reshape {self.shape} ({len(self.data)} values) to {shape}")
            shape = tuple(len(self.data) // known if d == -1 else d for d in shape)
        elif known != len(self.data):
            raise ValueError(f"Cannot reshape {self.shape} ({len(self.data)} values) to {shape}")

        out = tensor(self.data, shape=shape, children=(self,), op='reshape')

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]

        out._backward = _backward
        return out

    def transpose(self, dim0=-2, dim1=-1):
        # swaps two dims; the default swaps the last two, so a matrix becomes its transpose.
        # the values are written out again in the new order, so the output is a plain
        # tensor like any other. each output value is copied from one input position,
        # so backward sends each grad straight back to the position it came from.
        n = len(self.shape)
        for d in (dim0, dim1):
            if not -n <= d < n:
                raise IndexError(f"Dim {d} out of range for shape {self.shape}")
        dim0, dim1 = dim0 % n, dim1 % n

        shape = list(self.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        shape = tuple(shape)
        out_strides = self._compute_strides(shape)

        # src[i] is the position in self.data that output position i copies from
        src = []
        for i in range(len(self.data)):
            index = [(i // s) % d for s, d in zip(out_strides, shape)]
            index[dim0], index[dim1] = index[dim1], index[dim0]
            src.append(sum(k * s for k, s in zip(index, self.strides)))

        out = tensor([self.data[j] for j in src], shape=shape, children=(self,), op='transpose')

        def _backward():
            for i, j in enumerate(src):
                self.grad[j] += out.grad[i]

        out._backward = _backward
        return out

    @staticmethod
    def _broadcast_shape(a, b):
        # line the shapes up from the right. in each position the sizes must match or one
        # of them must be 1, and a missing dim counts as 1. (3, 1) and (4,) give (3, 4)
        n = max(len(a), len(b))
        a = (1,) * (n - len(a)) + tuple(a)
        b = (1,) * (n - len(b)) + tuple(b)
        shape = []
        for da, db in zip(a, b):
            if da != db and da != 1 and db != 1:
                raise ValueError(f"Shapes {a} and {b} can't be broadcast together")
            shape.append(db if da == 1 else da)
        return tuple(shape)

    def expand(self, target_shape):
        # repeats values along dims of size 1, and adds new dims on the left, until the
        # tensor has target_shape. many output positions copy the same input value, so
        # backward adds all their grads back into that one value.
        target_shape = tuple(target_shape)
        if self.shape == target_shape:
            return self
        if self.shape == (1,) and target_shape == ():
            return self.reshape(())
        extra = len(target_shape) - len(self.shape)
        if extra < 0 or any(d != 1 and d != t for d, t in zip(self.shape, target_shape[extra:])):
            raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")

        # an output index moves through self.data with self's stride, or not at all
        # along a dim that self repeats (size 1) or doesn't have (the new dims on the left)
        strides = [0] * extra + [0 if d == 1 else s for d, s in zip(self.shape, self.strides)]
        src = [0]
        for size, stride in zip(target_shape, strides):
            src = [j + k * stride for j in src for k in range(size)]

        out = tensor([self.data[j] for j in src], shape=target_shape, children=(self,), op='expand')

        def _backward():
            for i, j in enumerate(src):
                self.grad[j] += out.grad[i]

        out._backward = _backward
        return out

    def relu(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            # If negative make 0, else no change
            out_data[i] = 0.0 if self.data[i] < 0 else self.data[i]
        
        out = tensor(out_data, shape=self.shape, children=(self,), op="relu")
        def _backward():
            for i in range(len(self.data)):
                local_grad = 1.0 if out.data[i] > 0 else 0.0
                
                self.grad[i] += local_grad * out.grad[i]

        out._backward = _backward
        return out
    
    def __neg__(self): 
        return self * -1.0

    def __sub__(self, other): 
        return self + (-other)

    def __radd__(self, other): 
        return self + other

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other): 
        return self * other
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Only supporting int/float powers for now")
        if not float(other).is_integer() and any(v < 0 for v in self.data):
            raise ValueError(f"Negative base with power {other} gives a complex number")
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self.data[i] ** other
        out = tensor(out_data, shape=self.shape, children=(self,), op=f'**{other}')

        def _backward():
            for i in range(len(self.data)):
                local_grad = other * (self.data[i] ** (other - 1))
                self.grad[i] += local_grad * out.grad[i]

        out._backward = _backward
        return out
    
    def __truediv__(self, other): 
        return self * (other ** -1)

    def __rtruediv__(self, other): 
        return other * (self ** -1)
    

    @staticmethod
    def _taylor_log(x, terms=30):
        if x <= 0:
            raise ValueError(f"log is undefined for {x}")
        if x == float('inf'):
            return x

        # the series converges fast only near 1, so scale x into [0.5, 2]
        # and add back k * log(2)
        k = 0
        while x > 2.0:
            x /= 2.0
            k += 1
        while x < 0.5:
            x *= 2.0
            k -= 1

        z = (x - 1.0) / (x + 1.0)
        z_squared = z * z
        numerator = z
        result = 0.0

        for i in range(1, terms * 2, 2):
            result += numerator / i
            numerator *= z_squared

        return 2.0 * result + k * 0.6931471805599453

    def log(self):
        
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self._taylor_log(self.data[i] + 1e-8)
            
        out = tensor(out_data, shape=self.shape, children=(self,), op='log')

        def _backward():
            for i in range(len(self.data)):
                local_grad = 1.0 / (self.data[i] + 1e-8)
                self.grad[i] += local_grad * out.grad[i]
                
        out._backward = _backward
        return out
    
    # both helpers only ever call exp on a non-positive number, so nothing overflows
    @staticmethod
    def _sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + tensor._taylor_exp(-x))
        e = tensor._taylor_exp(x)
        return e / (1.0 + e)

    @staticmethod
    def _tanh(x):
        e = tensor._taylor_exp(-2.0 * abs(x))
        t = (1.0 - e) / (1.0 + e)
        return t if x >= 0 else -t

    def sigmoid(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self._sigmoid(self.data[i])

        out = tensor(out_data, shape=self.shape, children=(self,), op='sigmoid')

        def _backward():
            for i in range(len(self.data)):
                s = out.data[i]
                self.grad[i] += s * (1.0 - s) * out.grad[i]

        out._backward = _backward
        return out

    def tanh(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self._tanh(self.data[i])

        out = tensor(out_data, shape=self.shape, children=(self,), op='tanh')

        def _backward():
            for i in range(len(self.data)):
                local_grad = 1.0 - (out.data[i] ** 2)
                self.grad[i] += local_grad * out.grad[i]
                
        out._backward = _backward
        return out
    

    def leaky_relu(self, alpha=0.01):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            out_data[i] = self.data[i] if self.data[i] > 0 else (alpha * self.data[i])
        
        out = tensor(out_data, shape=self.shape, children=(self,), op="leaky_relu")
        def _backward():
            for i in range(len(self.data)):
                local_grad = 1.0 if self.data[i] > 0 else alpha
                self.grad[i] += local_grad * out.grad[i]
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdim=False):
        # axis=None adds up everything. with an axis, values that differ only along that
        # axis are added together, so that dim disappears (or stays as size 1 with keepdim).
        # every input value adds into one output value with weight 1, so backward copies
        # each output grad back to all the inputs that went into it.
        if axis is None:
            total = sum(self.data)
            shape = (1,) * len(self.shape) if keepdim else ()
            out = tensor([total], shape=shape, children=(self,), op='sum')

            def _backward():
                for i in range(len(self.data)):
                    self.grad[i] += out.grad[0]

            out._backward = _backward
            return out

        n = len(self.shape)
        if not -n <= axis < n:
            raise IndexError(f"Axis {axis} out of range for shape {self.shape}")
        axis = axis % n

        # flat data is laid out as [outer][size][inner]; summing over size leaves [outer][inner]
        outer = 1
        for d in self.shape[:axis]:
            outer *= d
        size = self.shape[axis]
        inner = 1
        for d in self.shape[axis + 1:]:
            inner *= d

        # dst[i] is the output position that input position i adds into
        dst = [o * inner + r for o in range(outer) for k in range(size) for r in range(inner)]
        out_data = [0.0] * (outer * inner)
        for i, j in enumerate(dst):
            out_data[j] += self.data[i]

        if keepdim:
            shape = self.shape[:axis] + (1,) + self.shape[axis + 1:]
        else:
            shape = self.shape[:axis] + self.shape[axis + 1:]
        out = tensor(out_data, shape=shape, children=(self,), op='sum')

        def _backward():
            for i, j in enumerate(dst):
                self.grad[i] += out.grad[j]

        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdim=False):
        # sum divided by how many values went into each sum. built from sum and *, so
        # backward comes for free: each input gets out.grad / count
        if axis is None:
            count = len(self.data)
        else:
            if not -len(self.shape) <= axis < len(self.shape):
                raise IndexError(f"Axis {axis} out of range for shape {self.shape}")
            count = self.shape[axis]
        return self.sum(axis, keepdim) * (1.0 / count)
    
    def var(self, axis=None, keepdim=False):
        # mean of the squared distance from the mean (divides by n, not n - 1).
        # the mean keeps its dim so it broadcasts back against self
        m = self.mean(axis, keepdim=True)
        return ((self - m) ** 2).mean(axis, keepdim)
    
    def std(self):
        # own op so a constant tensor gets gradient 0 (std is at its minimum there)
        # instead of hitting 0 ** -0.5 in pow's backward
        n = len(self.data)
        m = sum(self.data) / n
        s = (sum((v - m) ** 2 for v in self.data) / n) ** 0.5
        out = tensor([s], shape=(), children=(self,), op='std')

        def _backward():
            if s == 0.0:
                return
            for i in range(n):
                self.grad[i] += (self.data[i] - m) / (n * s) * out.grad[0]

        out._backward = _backward
        return out
    
    def softmax(self):
        # over the last dimension, so each row of a (B, C) batch sums to 1
        C = self.shape[-1] if self.shape else 1
        out_data = [0.0] * len(self.data)
        for start in range(0, len(self.data), C):
            row = self.data[start:start + C]
            max_val = max(row)
            exp_vals = [self._taylor_exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            for j in range(C):
                out_data[start + j] = exp_vals[j] / sum_exp

        out = tensor(out_data, shape=self.shape, children=(self,), op='softmax')

        def _backward():
            for start in range(0, len(self.data), C):
                dot = 0.0
                for j in range(C):
                    dot += out.data[start + j] * out.grad[start + j]
                for j in range(C):
                    s = out.data[start + j]
                    self.grad[start + j] += s * (out.grad[start + j] - dot)

        out._backward = _backward
        return out
    
    def mse_loss(self, target):
        if self.shape != target.shape:
            raise ValueError(f"Shape mismatch in mse_loss: {self.shape} vs {target.shape}")
        
        diff_data = [self.data[i] - target.data[i] for i in range(len(self.data))]
        squared = [d * d for d in diff_data]
        loss_val = sum(squared) / len(self.data)
        
        out = tensor([loss_val], shape=(), children=(self, target), op='mse_loss')
        
        def _backward():
            scale = 2.0 / len(self.data)
            for i in range(len(self.data)):
                grad_contrib = scale * (self.data[i] - target.data[i]) * out.grad[0]
                self.grad[i] += grad_contrib
        
        out._backward = _backward
        return out
    
    def cross_entropy_loss(self, target):
        if self.shape != target.shape:
            raise ValueError(f"Shape mismatch in cross_entropy_loss: {self.shape} vs {target.shape}")
        
        # softmax over the last dimension, loss averaged over rows.
        # each target row should be one-hot or sum to 1
        C = self.shape[-1] if self.shape else 1
        rows = len(self.data) // C
        softmax_vals = [0.0] * len(self.data)

        loss_val = 0.0
        for start in range(0, len(self.data), C):
            row = self.data[start:start + C]
            max_val = max(row)
            exp_vals = [self._taylor_exp(v - max_val) for v in row]
            sum_exp = sum(exp_vals)
            log_sum_exp = self._taylor_log(sum_exp)
            for j in range(C):
                softmax_vals[start + j] = exp_vals[j] / sum_exp
                # log softmax = x - max - log(sum(exp(x - max)))
                loss_val -= target.data[start + j] * (row[j] - max_val - log_sum_exp)
        loss_val /= rows

        out = tensor([loss_val], shape=(), children=(self, target), op='cross_entropy')

        def _backward():
            for i in range(len(self.data)):
                grad_contrib = (softmax_vals[i] - target.data[i]) / rows * out.grad[0]
                self.grad[i] += grad_contrib
        
        out._backward = _backward
        return out
    
    def gelu(self):
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k = 0.7978845608028654
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            x = self.data[i]
            t = self._tanh(k * (x + 0.044715 * x * x * x))
            out_data[i] = 0.5 * x * (1.0 + t)

        out = tensor(out_data, shape=self.shape, children=(self,), op='gelu')

        def _backward():
            for i in range(len(self.data)):
                x = self.data[i]
                t = self._tanh(k * (x + 0.044715 * x * x * x))
                du_dx = k * (1.0 + 3 * 0.044715 * x * x)
                local_grad = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
                self.grad[i] += local_grad * out.grad[i]
                
        out._backward = _backward
        return out


def gradcheck(f, *inputs, eps=1e-6, atol=1e-5, rtol=1e-4):
    # checks backward() against central finite differences: nudge each input value
    # up and down by eps and measure how much the output moves, (f(x+eps) - f(x-eps)) / 2eps.
    # f must return a single value, like a loss. returns True or raises at the first mismatch
    for x in inputs:
        x.grad = [0.0] * len(x.data)
    out = f(*inputs)
    if len(out.data) != 1:
        raise ValueError(f"gradcheck needs f to return a single value, got shape {out.shape}")
    out.backward()

    for n, x in enumerate(inputs):
        for i in range(len(x.data)):
            old = x.data[i]
            x.data[i] = old + eps
            up = f(*inputs).data[0]
            x.data[i] = old - eps
            down = f(*inputs).data[0]
            x.data[i] = old

            numeric = (up - down) / (2 * eps)
            analytic = x.grad[i]
            if abs(analytic - numeric) > atol + rtol * abs(numeric):
                raise ValueError(f"Gradient mismatch at input {n}, index {i}: "
                                 f"backward gave {analytic}, finite differences gave {numeric}")
    return True
