class tensor:
    def __init__(self, data, shape=None, children=(), op=''):
        if shape is None:
            flat_data, inferred_shape = self._parse_input(data)
            self.data = flat_data
            self.shape = inferred_shape
        else:
            self.data = list(data)
            self.shape = shape
            
        self.grad = [0.0] * len(self.data)
        
        self.strides = self._compute_strides(self.shape)
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None

    @staticmethod
    def _taylor_exp(x, terms=15):
        result = 1.0
        current_term = 1.0
        
        for i in range(1, terms):
            current_term *= (x / i)
            result += current_term
            
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
            return list(data), (len(data),)



    def __repr__(self):
        return f"tensor(data={self.data}, shape={self.shape})"



    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = tensor([other] * len(self.data), shape=self.shape)
        if isinstance(other, tensor) and self.shape != other.shape:
            other = other.expand(self.shape)
        
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
            other = other.expand(self.shape)

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

    def backward(self):
        topo= []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
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
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        return self.data[flat_index]

    def expand(self, target_shape):
        if self.shape == target_shape:
            return self
        target_len = 1
        for dim in target_shape:
            target_len *= dim

        # Expand a scalar
        if self.shape == (1,) or self.shape == ():
            # treat empty-shape scalar and (1,) as scalar
            out_data = [self.data[0]] * target_len
            out = tensor(out_data, shape=target_shape, children=(self,), op='expand')
            def _backward():
                s = 0.0
                for v in out.grad:
                    s += v
                self.grad[0] += s
            out._backward = _backward
            return out

        # Expand a bias row (1, N) to (M, N)
        if len(self.shape) == 2 and self.shape[0] == 1 and len(target_shape) == 2 and self.shape[1] == target_shape[1]:
            M, N = target_shape
            out_data = self.data * M
            out = tensor(out_data, shape=target_shape, children=(self,), op='expand')
            def _backward():
                for i in range(M * N):
                    self.grad[i % N] += out.grad[i]
            out._backward = _backward
            return out
        raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")
    

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
            return -1e9 
            
        z = (x - 1.0) / (x + 1.0)
        z_squared = z * z
        numerator = z
        result = 0.0
        
        for i in range(1, terms * 2, 2):
            result += numerator / i
            numerator *= z_squared
            
        return 2.0 * result

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
    
    def sigmoid(self):
        return ((self * -1.0).exp() + 1.0) ** -1.0
    
    def tanh(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            e2x = self._taylor_exp(2.0 * self.data[i])
            out_data[i] = (e2x - 1.0) / (e2x + 1.0)
            
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

    def sum(self):
        total = sum(self.data)
        out = tensor([total], shape=(), children=(self,), op='sum')

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[0]

        out._backward = _backward
        return out
    
    def mean(self):
        return (self.sum() * (1.0 / len(self.data)))
    
    def var(self):
        m = self.mean()
        return ((self - m) ** 2).mean()
    
    def std(self):
        return self.var() ** 0.5
    
    def softmax(self):
        exponents = self.exp()
        return exponents / exponents.sum()
    
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
        
        max_val = max(self.data)
        exp_vals = [self._taylor_exp(self.data[i] - max_val) for i in range(len(self.data))]
        sum_exp = sum(exp_vals)
        softmax_vals = [e / sum_exp for e in exp_vals]
        
        loss_val = 0.0
        for i in range(len(self.data)):
            if softmax_vals[i] > 0:
                loss_val -= target.data[i] * self._taylor_log(softmax_vals[i] + 1e-8)
        
        out = tensor([loss_val], shape=(), children=(self, target), op='cross_entropy')
        
        def _backward():
            max_val = max(self.data)
            exp_vals = [self._taylor_exp(self.data[i] - max_val) for i in range(len(self.data))]
            sum_exp = sum(exp_vals)
            softmax_vals = [e / sum_exp for e in exp_vals]
            
            for i in range(len(self.data)):
                grad_contrib = (softmax_vals[i] - target.data[i]) * out.grad[0]
                self.grad[i] += grad_contrib
        
        out._backward = _backward
        return out
    
    def gelu(self):
        out_data = [0.0] * len(self.data)
        for i in range(len(self.data)):
            x = self.data[i]
            cdf = 0.5 * (1.0 + self._taylor_exp(0.7978845608 * x) / (self._taylor_exp(0.7978845608 * x) + 1.0))
            out_data[i] = x * cdf
            
        out = tensor(out_data, shape=self.shape, children=(self,), op='gelu')

        def _backward():
            for i in range(len(self.data)):
                x = self.data[i]
                cdf = 0.5 * (1.0 + self._taylor_exp(0.7978845608 * x) / (self._taylor_exp(0.7978845608 * x) + 1.0))
                local_grad = cdf + x * 0.3989422804 * self._taylor_exp(0.7978845608 * x) / ((self._taylor_exp(0.7978845608 * x) + 1.0) ** 2)
                self.grad[i] += local_grad * out.grad[i]
                
        out._backward = _backward
        return out