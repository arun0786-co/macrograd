class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr 

    def step(self):
        for p in self.parameters:
            for i in range(len(p.data)):
              
                p.data[i] -= self.lr * p.grad[i]

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0 
        
       
        self.m = [[0.0] * len(p.data) for p in self.parameters]
        self.v = [[0.0] * len(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            for i in range(len(p.data)):
                p.grad[i] = 0.0

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